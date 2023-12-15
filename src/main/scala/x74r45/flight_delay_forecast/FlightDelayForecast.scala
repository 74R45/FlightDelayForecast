package x74r45.flight_delay_forecast

import org.apache.spark.sql._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{OneHotEncoder, PolynomialExpansion, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression

import java.sql.Timestamp
import java.util.Calendar

object FlightDelayForecast {
  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark: SparkSession =
    SparkSession
      .builder()
      .appName("Flight Delay Forecast")
      .master("local")
      .getOrCreate()

  // For implicit conversions
  import spark.implicits._

  private val GenerateOutput = false

  def main(args: Array[String]): Unit = {
    val initDS = read("src/main/resources/x74r45/flight_delay_forecast/flights_jan2019.csv")
    if (GenerateOutput) initDS.summary().show()

    val flightsDS = processData(initDS).persist()
    flightsDS.createOrReplaceTempView("flights")
    if (GenerateOutput) flightsDS.show(8)

    if (GenerateOutput) generatePlots(flightsDS)

    trainModel1(flightsDS, "DL", 10397)
    trainModel2(flightsDS, "WN")

    spark.close()
  }

  def read(path: String): Dataset[InitFlight] =
    spark.read.options(Map("header" -> "true", "inferSchema" -> "true")).csv(path).as[InitFlight]

  def generatePlots(ds: Dataset[Flight]): Unit = {
    //DataAnalyser.plotAirlinesDistr(getAirlinesDistr(ds))
    DataAnalyser.plotAirlineMeanDelays(getAirlineMeanDelays(ds))
    DataAnalyser.plotDelaysPerAirline(getAirlineDelays("flights"))
    DataAnalyser.plotAirportsPerAirline(getAirportsPerAirline(ds))
    DataAnalyser.plotAirportMeanDelays(getAirportMeanDelays(ds))
    //DataAnalyser.plotAirportAirlineMeanDelays(getAirportAirlineMeanDelays(ds))
    DataAnalyser.plotTimeOfDayDelays(getDelaysPerTimeOfDay(ds), getMeanDelaysPerTimeOfDay(ds))
  }

  def processData(ds: Dataset[InitFlight]): Dataset[Flight] = {
    val getMinutes = (time0: String) => {
      val time = f"${time0.toInt}%04d"
      assert(time.length == 4)
      val hrs = time.substring(0, 2).toInt
      val mins = time.substring(2).toInt
      hrs * 60 + mins
    }

    val getSchedDep = (flight: InitFlight) => {
      val mins0 = getMinutes(flight.depTime) - flight.depDelay.toInt
      val mins = if (mins0 < 0) mins0 + 24 * 60 else mins0

      val calendar = new Calendar.Builder()
        .setDate(flight.year, flight.month, flight.day)
        .setTimeOfDay(mins / 60, mins % 60, 0)
        .build()
      new Timestamp(calendar.getTimeInMillis)
    }

    ds.na.drop().as[InitFlight].map(fl => Flight(
      getSchedDep(fl),
      fl.airline,
      fl.origin,
      getMinutes(fl.depTime),
      fl.depDelay
    ))
  }

  def getAirlinesDistr(ds: Dataset[Flight]): Seq[(String, Double)] = {
    val counts = ds.select($"airline").groupBy($"airline").count()
      .orderBy($"count".desc).collect().map {
      case Row(airline: String, count: Long) => (airline, count)
    }
    val total = counts.foldLeft(0L){case (total, (_, count)) => total + count}
    counts.map{case (airline, count) => (airline, count.toDouble / total)}
  }

  def getAirlineMeanDelays(ds: Dataset[Flight]): Seq[(String, Double)] = {
    ds.select($"airline", $"depDelay")
      .groupBy($"airline").avg("depDelay").collect().map {
      case Row(airline: String, avg: Double) => (airline, avg)
    }
  }

  def getAirlineDelays(ds: String): Seq[(String, Seq[Double])] = {
    spark.sql(
      s"""SELECT airline, count(CASE WHEN depDelay < 5 THEN 1 END) AS onTime,
         |count(CASE WHEN depDelay >= 5 AND depDelay < 60 THEN 1 END) AS smDelay,
         |count(CASE WHEN depDelay >= 60 THEN 1 END) AS lgDelay
         |FROM $ds
         |GROUP BY airline
         |""".stripMargin).collect().map {
      case Row(airline: String, onTime: Long, smDelay: Long, lgDelay: Long) =>
        (airline, Seq(onTime.toDouble, smDelay.toDouble, lgDelay.toDouble))
    }
  }

  def getAirportsPerAirline(ds: Dataset[Flight]): Seq[(String, Double)] = {
    ds.select($"airline", $"origin").distinct().groupBy($"airline").count()
      .collect().map {
      case Row(airline: String, count: Long) => (airline, count.toDouble)
    }
  }

  def getAirportAirlineMeanDelays(ds: Dataset[Flight]): Seq[(String, String, Double)] = {
    ds.select($"airline", $"origin", $"depDelay").groupBy($"airline", $"origin").avg("depDelay")
      .collect().map {
      case Row(airline: String, origin: Int, avg: Double) => (airline, origin.toString, avg)
    }
  }

  def getAirportMeanDelays(ds: Dataset[Flight]): Seq[(String, Double)] = {
    ds.select($"origin", $"depDelay")
      .groupBy($"origin").avg("depDelay").collect().map {
      case Row(origin: Int, avg: Double) => (origin.toString, avg)
    }
  }

  def getMeanDelaysPerTimeOfDay(ds: Dataset[Flight]): Seq[(Int, Int, Double)] = {
    val mapper: Flight => TimeDelay = fl => {
      val cal = Calendar.getInstance()
      cal.setTime(fl.schedDep)
      val hrs = cal.get(Calendar.HOUR_OF_DAY)
      val mins = cal.get(Calendar.MINUTE) / 10 * 10
      TimeDelay(hrs, mins, fl.depDelay)
    }

    ds.map(mapper).groupBy($"hours", $"minutes").avg("depDelay")
      .orderBy($"hours", $"minutes").collect().map {
      case Row(hours: Int, minutes: Int, avg: Double) => (hours, minutes, avg)
    }
  }

  def getDelaysPerTimeOfDay(ds: Dataset[Flight]): Seq[(Int, Int, Double)] = {
    val mapper: Flight => TimeDelay = fl => {
      val cal = Calendar.getInstance()
      cal.setTime(fl.schedDep)
      val hrs = cal.get(Calendar.HOUR_OF_DAY)
      val mins = cal.get(Calendar.MINUTE)
      TimeDelay(hrs, mins, fl.depDelay)
    }

    val filter: Flight => Boolean = fl => {
      val cal = Calendar.getInstance()
      cal.setTime(fl.schedDep)
      cal.get(Calendar.DAY_OF_MONTH) < 8
    }

    ds.filter(filter).map(mapper).toDF.orderBy($"hours", $"minutes").collect().map {
      case Row(hours: Int, minutes: Int, depDelay: Double) => (hours, minutes, depDelay)
    }
  }

  def popularAirlineAirportCombos(ds: Dataset[Flight]): Unit = {
    ds.groupBy($"airline", $"origin").count().orderBy($"count".desc).show()
  }

  def trainModel1(ds: Dataset[Flight], airline: String, airport: Int): Unit = {
    val mapper: Flight => FlightModel1 = fl => {
      val cal = Calendar.getInstance()
      cal.setTime(fl.schedDep)
      val hrs = cal.get(Calendar.HOUR_OF_DAY)
      val mins = cal.get(Calendar.MINUTE)
      FlightModel1(hrs*60 + mins, fl.depDelay)
    }

    val ds2 = new VectorAssembler()
      .setInputCols(Array("schedDepTime"))
      .setOutputCol("features")
      .transform(ds.filter(fl => fl.airline == airline && fl.origin == airport && fl.depDelay < 60).map(mapper))

    for (deg <- 1 to 3) {
      println(s"Polynomial degree: $deg")

      val ds2Poly = new PolynomialExpansion()
        .setDegree(deg)
        .setInputCol("features")
        .setOutputCol("polyFeatures")
        .transform(ds2)

      val Array(trainingDS, testDS) = ds2Poly.randomSplit(Array(0.7, 0.3))

      val lr = new LinearRegression()
        .setLabelCol("depDelay")
        .setFeaturesCol("polyFeatures")
//        .setTol(0.0001)
//        .setRegParam(0.3)
//        .setElasticNetParam(0.8)

      val model = lr.fit(trainingDS)

      println("Test DS:")
      model.transform(testDS).show(5)

      // Print the coefficients and intercept for linear regression
      println(s"Coefficients: ${model.coefficients}\n Intercept: ${model.intercept}")

      // Summarize the model over the training set and print out some metrics
      val trainingSummary = model.summary
      println(s"numIterations: ${trainingSummary.totalIterations}")
      println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
      val wrongGuesses = trainingSummary.residuals.as[Double].filter(math.abs(_) > 15d).count().toDouble / trainingSummary.residuals.count()
      println(s"Error > 15 min: ${wrongGuesses * 100}%")
      println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
      println(s"MSE: ${trainingSummary.meanSquaredError}")
      println(s"r2: ${trainingSummary.r2}")
    }
  }

  def trainModel2(ds: Dataset[Flight], airline: String): Unit = {
    val mapper: Flight => FlightModel2 = fl => {
      val cal = Calendar.getInstance()
      cal.setTime(fl.schedDep)
      val hrs = cal.get(Calendar.HOUR_OF_DAY)
      val mins = cal.get(Calendar.MINUTE)
      FlightModel2(hrs*60 + mins, fl.origin, fl.depDelay)
    }
    val ds2 = ds.filter(fl => fl.airline == airline && fl.depDelay < 60).map(mapper)

    val ds2Indexed = new StringIndexer()
      .setInputCol("origin")
      .setOutputCol("originIndexed")
      .fit(ds2)
      .transform(ds2)

    val ds2Encoded = new OneHotEncoder()
      .setInputCol("originIndexed")
      .setOutputCol("originEncoded")
      .fit(ds2Indexed)
      .transform(ds2Indexed)

    val ds2Vec = new VectorAssembler()
      .setInputCols(Array("schedDepTime", "originEncoded"))
      .setOutputCol("features")
      .transform(ds2Encoded)

    val Array(trainingDS, testDS) = ds2Vec.randomSplit(Array(0.7, 0.3))

    val lr = new LinearRegression()
      .setLabelCol("depDelay")
      .setFeaturesCol("features")
      //.setRegParam(0.3)
      .setElasticNetParam(0.5)

    val model = lr.fit(trainingDS)

    println("Test DS:")
    model.transform(testDS).show(5)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${model.coefficients}\n Intercept: ${model.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    val wrongGuesses = trainingSummary.residuals.as[Double].filter(math.abs(_) > 15d).count().toDouble / trainingSummary.residuals.count()
    println(s"Error > 15 min: ${wrongGuesses * 100}%")
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
}

case class InitFlight(
  year: Int,
  month: Int,
  day: Int,
  airline: String,
  origin: Int,
  dest: Int,
  depTime: String,
  depDelay: Double,
  arrTime: String,
  arrDelay: Double,
  distance: Double
)

case class Flight(
  schedDep: Timestamp,
  airline: String,
  origin: Int,
  depTime: Int,
  depDelay: Double
)

case class TimeDelay(hours: Int, minutes: Int, depDelay: Double)
case class FlightModel1(schedDepTime: Int, depDelay: Double)
case class FlightModel2(schedDepTime: Int, origin: Int, depDelay: Double)