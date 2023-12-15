name := "FlightDelayForecast"

version := "0.1"

scalaVersion := "2.12.8"
scalacOptions ++= Seq("-language:implicitConversions", "-deprecation")
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.1.1",
  "org.apache.spark" %% "spark-sql" % "3.1.1",
  "org.apache.spark" %% "spark-mllib" % "3.1.1",
  "com.cibo" %% "evilplot" % "0.7.1"
)

resolvers += Resolver.bintrayRepo("cibotech", "public")