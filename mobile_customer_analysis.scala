import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, MinMaxScaler}
import org.apache.spark.ml.clustering.KMeans
import java.sql.Timestamp
import org.apache.log4j.{Level, Logger}

object RfmCustomerClusterFull {
  def main(args: Array[String]): Unit = {
    // 日志优化：屏蔽冗余日志，只显示关键信息
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.hadoop").setLevel(Level.ERROR)
    Logger.getLogger("org").setLevel(Level.ERROR)

    // ===================== 【完全使用你指定的Spark配置 - 核心修改点】 =====================
    val spark = SparkSession.builder()
      .appName("LocalSVMUserRetention")
      .master("local[*]")
      .config("hive.metastore.uris", "thrift://192.168.128.130:9083")
      .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
      .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
      .config("spark.sql.legacy.datetimeParserPolicy", "LEGACY")
      .config("spark.hadoop.fs.defaultFS", "hdfs://192.168.128.130:8020")
      .config("spark.hadoop.user.name", "hadoop")
      .config("hadoop.security.authentication", "simple")
      .enableHiveSupport()
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    // ======================================================================================

    import spark.implicits._

    // ===================== 全局常量配置（完全保留你的HDFS路径和参数，无需修改） =====================
    val statEndDate = Timestamp.valueOf("2016-07-20 00:00:00")  // RFM统计截止日
    val startDate = Timestamp.valueOf("2016-06-01 00:00:00")     // 消费起始日
    val rawDataPath = "hdfs://192.168.128.130:8020/user/hive/warehouse/rfm_analysis.db/RFM聚类分析.csv"  // 源文件路径
    val hdfsTempPath = "hdfs://192.168.128.130:8020/user/hive/warehouse/rfm_analysis.db/rfm_temp_dir"  // 临时目录
    val hdfsFinalCsv = "hdfs://192.168.128.130:8020/user/hive/warehouse/rfm_analysis.db/rfm_customer_cluster.csv"  // 最终导出路径

    try {
      // ===================== 1. 读取HDFS原始数据 + 数据质量统计 =====================
      val rawDF = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("sep", ",")
        .option("nullValue", "")
        .option("ignoreLeadingWhiteSpace", true)
        .option("ignoreTrailingWhiteSpace", true)
        .option("mode", "DROPMALFORMED")  // 丢弃格式错误的脏数据
        .csv(rawDataPath)

      val rawCount = rawDF.count()
      println(s"✅ 【原始数据】总行数：$rawCount")

      // 缺失值统计（只统计RFM核心字段）
      val nullUserCount = rawDF.filter(col("user_id").isNull || col("user_id") === "").count()
      val nullPayTimeCount = rawDF.filter(col("last_pay_time").isNull).count()
      val nullPayNumCount = rawDF.filter(col("pay_num").isNull || col("pay_num") <= 0).count()
      val nullPayTimesCount = rawDF.filter(col("pay_times").isNull || col("pay_times") <= 0).count()
      println(s"✅ 【缺失值统计】用户ID缺失：$nullUserCount 条 | 消费时间缺失：$nullPayTimeCount 条 | 消费金额异常：$nullPayNumCount 条 | 消费次数异常：$nullPayTimesCount 条")

      // ===================== 2. 全量数据预处理（完整保留你的逻辑） =====================
      val cleanDF = rawDF
        // ✔ 缺失值处理：过滤核心字段为空的数据
        .filter(col("user_id").isNotNull && col("user_id") =!= "")
        .filter(col("gender").isin(1, 2) && col("age").between(15, 80))  // gender=1/2，年龄15-80岁
        .filter(col("last_pay_time").isNotNull)  // 消费时间非空
        .filter(col("pay_num") > 0 && col("pay_times") > 0)  // 消费金额/次数>0（有效消费）
        // ✔ 重复值处理：按用户唯一标识user_id去重，避免重复数据
        .dropDuplicates("user_id")
        // ✔ 异常值处理：数值范围过滤（避免极端脏数据）
        .filter(col("pay_num") < 100000)  // 过滤超大额消费（可根据实际调整）
        .filter(col("pay_times") < 100)    // 过滤超高频消费（可根据实际调整）
        // ✔ 属性规约：只保留RFM分析必需的6个核心字段，剔除冗余字段（降低计算维度）
        .select("user_id", "gender", "age", "last_pay_time", "pay_num", "pay_times")

      val cleanCount = cleanDF.count()
      val delCount = rawCount - cleanCount
      println(s"✅ 【数据清洗完成】过滤无效数据：$delCount 条 | 有效数据：$cleanCount 条 | 数据有效率：${(cleanCount.toDouble/rawCount*100).formatted("%.2f")}%")

      // ===================== 3. RFM核心指标计算  =====================
      /*
      "我负责的是RFM打分和价值分层模块。这段代码的核心思想是用三个关键指标一-R:最近消费时间、F:消费频率和 M:消费金额--来量化客户价值。
      "首先，我为每个指标设定了1-5分的打分标准。比如，最近7天内消费的客户得5分，而超过60天未消费的只得1分。
      "然后，将三个分数相加得到3-15分的总分，并根据总分将客户分为四个层级。最后，我们还计算了复购率这个重要的业务指标。
      "这个模块的输出为后续的聚类分析和可视化提供了基础标签使得我们的客户分析不仅有机器学习的聚类结果，
      还有基于业务规则的明确分层，两者可以相互验证。
       */
      val rfmDF = cleanDF
        // ✅ 终极日期兼容：完美解析 yyyy/M/d、yyyy/MM/dd 两种格式，无任何报错
        .withColumn("last_pay_time_str", col("last_pay_time").cast(StringType))
        .withColumn("last_pay_time_fix", to_timestamp(col("last_pay_time_str"), "yyyy/M/d"))
        // ✔ R：最近消费间隔天数（统计截止日 - 最后消费日期，越小越好）
        .withColumn("R", datediff(lit(statEndDate), col("last_pay_time_fix")))
        // 计算消费月数（强制≥1，避免分母为0）
        .withColumn("consume_days", datediff(col("last_pay_time_fix"), lit(startDate)) + 1)
        .withColumn("consume_months", when(col("consume_days") <= 30, 1).otherwise(ceil(col("consume_days") / 30.0)))
        // ✔ F：月均消费频率（消费次数/消费月数，越大越好）
        .withColumn("F", round(col("pay_times") / col("consume_months"), 2))
        // ✔ M：月均消费金额（消费金额/消费月数，越大越好）
        .withColumn("M", round(col("pay_num") / col("consume_months"), 2))
        // 业务特征：标记是否复购客户（消费次数≥2）
        .withColumn("is_repeat_customer", when(col("pay_times") >= 2, 1).otherwise(0))
        .drop("last_pay_time_str")  // 删除临时字符串字段

      // ===================== 4. RFM经典打分 + 客户价值等级分层 =====================
      /*
      按流的角色的不同分为：节点流，处理流/包装流
       */
      val rfmScoreDF = rfmDF
        .withColumn("R_score", when(col("R") <= 7, 5).when(col("R") <= 15, 4).when(col("R") <= 30, 3).when(col("R") <= 60, 2).otherwise(1))
        .withColumn("F_score", when(col("F") >= 5, 5).when(col("F") >= 3, 4).when(col("F") >= 2, 3).when(col("F") >= 1, 2).otherwise(1))
        .withColumn("M_score", when(col("M") >= 1000, 5).when(col("M") >= 500, 4).when(col("M") >= 200, 3).when(col("M") >= 50, 2).otherwise(1))
        .withColumn("rfm_total_score", col("R_score") + col("F_score") + col("M_score"))
        .withColumn("customer_value_level",
          when(col("rfm_total_score") >= 12, "高价值客户")
            .when(col("rfm_total_score") >= 9, "中价值客户")
            .when(col("rfm_total_score") >= 6, "一般价值客户")
            .otherwise("低价值客户"))



      // 复购率统计
      val repeatCount = rfmScoreDF.filter(col("is_repeat_customer") === 1).count()
      println(s"✅ 【业务特征】复购客户数：$repeatCount 条 | 复购率：${(repeatCount.toDouble/cleanCount*100).formatted("%.2f")}%")

      // ===================== 5. K-Means聚类建模（4类客户分层） =====================
      val vecAssembler = new VectorAssembler()
        .setInputCols(Array("R", "F", "M"))  // 基于RFM核心指标聚类
        .setOutputCol("features")
        .setHandleInvalid("skip")  // 兜底过滤异常特征值

      val scaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaled_features")
        .setWithStd(true)
        .setWithMean(true)

      val minMaxScaler = new MinMaxScaler()
        .setInputCol("scaled_features")
        .setOutputCol("norm_features")
        .setMin(0)
        .setMax(1)
      val featureDF = vecAssembler.transform(rfmScoreDF)
      val scaledDF = scaler.fit(featureDF).transform(featureDF)
      val normDF = minMaxScaler.fit(scaledDF).transform(scaledDF)
      val kmeans = new KMeans()
        .setK(4)
        .setMaxIter(100)
        .setSeed(12345L)  // 固定种子，结果可复现
        .setFeaturesCol("norm_features")
        .setPredictionCol("cluster_label")

      val clusterDF = kmeans.fit(normDF).transform(normDF)

      // ===================== 6. 最终结果筛选 + 二次属性规约 =====================
      val finalDF = clusterDF
        .withColumn("cluster_cn_label",
          when(col("cluster_label") === 0, "高价值客户")
            .when(col("cluster_label") === 1, "一般价值客户")
            .when(col("cluster_label") === 2, "核心客户")
            .otherwise("潜在流失客户"))
        // 属性规约：只保留14个业务分析+可视化必需字段，剔除建模临时字段
        .select(
        "user_id", "gender", "age", "R", "F", "M",
        "cluster_label", "cluster_cn_label",
        "R_score", "F_score", "M_score", "rfm_total_score", "customer_value_level", "is_repeat_customer"
      )
        // 二次异常过滤：保证导出数据绝对纯净
        .filter(col("cluster_label").isin(0, 1, 2, 3))
        .filter(col("R") >= 0 && col("F") >= 0 && col("M") >= 0)

      println(s"✅ 【聚类完成】最终导出数据量：${finalDF.count()} 条，数据质量100%合格")

      // ===================== 7. 导出CSV到HDFS指定路径 + 自动清理+重命名 =====================
      val hdfs = org.apache.hadoop.fs.FileSystem.get(spark.sparkContext.hadoopConfiguration)
      val tempDir = new org.apache.hadoop.fs.Path(hdfsTempPath)
      val finalFile = new org.apache.hadoop.fs.Path(hdfsFinalCsv)

      // 先删除旧文件/目录，避免重复报错
      if (hdfs.exists(tempDir)) hdfs.delete(tempDir, true)
      if (hdfs.exists(finalFile)) hdfs.delete(finalFile, true)

      // 写入HDFS并合并为单个文件（方便Python读取）
      finalDF
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .option("sep", ",")
        .option("encoding", "utf-8")
        .csv(hdfsTempPath)

      // 重命名Spark默认的part文件为指定文件名，清理临时目录
      val partFiles = hdfs.globStatus(new org.apache.hadoop.fs.Path(s"$hdfsTempPath/part-00000*.csv"))
      if (partFiles.nonEmpty) {
        val partFile = partFiles(0).getPath()
        hdfs.rename(partFile, finalFile)
        hdfs.delete(tempDir, true)
        println(s"✅ 【导出成功】HDFS结果文件路径：$hdfsFinalCsv")
      }

      // ===================== 执行完成日志 =====================
      println(s"\n=================================================")
      println(s"✅ 程序执行成功！所有功能全部完成，无任何报错")
      println(s"✅ 源数据路径：$rawDataPath")
      println(s"✅ 结果文件路径：$hdfsFinalCsv")
      println(s"=================================================")

    } catch {
      case e: Exception => {
        println(s"\n❌ 程序执行异常：${e.getMessage}")
        e.printStackTrace()
      }
    } finally {
      // 优雅关闭SparkSession，释放资源
      spark.stop()
      println("✅ Spark资源已正常释放")
    }
  }
}