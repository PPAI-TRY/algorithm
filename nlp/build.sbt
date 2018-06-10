name := "nlp"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.0"


assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs @ _*) => MergeStrategy.last
  case PathList("javax", "activation", xs @ _*) => MergeStrategy.last
  case PathList("org", "apache", xs @ _*) => MergeStrategy.last
  case "about.html" => MergeStrategy.rename
  case "META-INF/ECLIPSEF.RSA" => MergeStrategy.last
  case "META-INF/mailcap" => MergeStrategy.last
  case "META-INF/mimetypes.default" => MergeStrategy.last
  case "plugin.properties" => MergeStrategy.last
  case "log4j.properties" => MergeStrategy.last
  case x => {
    val old = (assemblyMergeStrategy in assembly).value
    old(x)
  }
}

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

assemblyOption in assembly := (assemblyOption in assembly).value.copy(cacheUnzip = false)

assemblyOption in assembly := (assemblyOption in assembly).value.copy(cacheOutput = false)

