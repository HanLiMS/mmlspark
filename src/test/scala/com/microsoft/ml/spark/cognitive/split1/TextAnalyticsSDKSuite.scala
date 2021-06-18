package com.microsoft.ml.spark.cognitive.split1

import com.azure.ai.textanalytics.models.TextAnalyticsRequestOptions
import com.microsoft.ml.spark.Secrets
import com.microsoft.ml.spark.cognitive._
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.TransformerFuzzing
import org.apache.spark.ml.param.DataFrameEquality
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col

class TextAnalyticsSDKSuite extends TestBase with DataFrameEquality with TextKey {
  import spark.implicits._

  lazy val df: DataFrame = Seq(
    "Hello World",
    "Bonjour tout le monde",
    "La carretera estaba atascada. Había mucho tráfico el día de ayer.",
    ":) :( :D"
  ).toDF("text2")

  val options: Option[TextAnalyticsRequestOptions] = Some(new TextAnalyticsRequestOptions()
    .setIncludeStatistics(true))

  lazy val detector: TextAnalyticsLanguageDetection = new TextAnalyticsLanguageDetection(options)
    .setSubscriptionKey(textKey)
    .setEndpoint("https://eastus.api.cognitive.microsoft.com/")
    .setInputCol("text2")

  test("Basic Usage") {
    val replies = detector.transform(df)
      .select("name", "iso6391Name")
      .collect()

    assert(replies(0).getString(0) == "English" && replies(2).getString(0) == "Spanish")
    assert(replies(0).getString(1) == "en" && replies(2).getString(1) == "es")
  }
}
class TextSentimentSuiteV4 extends TestBase with DataFrameEquality with TextKey {

  import spark.implicits._

  lazy val df: DataFrame = Seq(
    ("Hello world. This is some input text that I love."),
    ("It is always raining in Seattle. So bad. I hate the rainy season."),
  ).toDF("text")

  val options: Option[TextAnalyticsRequestOptions] = Some(new TextAnalyticsRequestOptions()
    .setIncludeStatistics(true))

  lazy val detector: TextSentimentV4 = new TextSentimentV4(options)
    .setSubscriptionKey("29e438c2cc004ca2a49c6fd10a4f65fe")
    .setEndpoint("https://test-v2-endpoints.cognitiveservices.azure.com/")
    .setInputCol("text")

  test("foo") {
    detector.transform(df).printSchema()
  }
  test("Basic Usage") {
    val replies = detector.transform(df)
      .select("textSentiment")
      .collect()
    assert(replies(0).getString(0) == "positive" && replies(1).getString(0) == "negative")
  }

  lazy val extractor: TextAnalyticsKeyphraseExtraction = new TextAnalyticsKeyphraseExtraction(options)
    .setSubscriptionKey("29e438c2cc004ca2a49c6fd10a4f65fe")
    .setEndpoint("https://test-v2-endpoints.cognitiveservices.azure.com/")
    .setInputCol("text")

  test("kpe foo") {
    extractor.transform(df).printSchema()
  }


  test("Basic KPE Usage") {
    val replies = extractor.transform(df)
      .select("keyPhrases")
      .collect()

    println(replies)
    assert(replies(0).getSeq[String](0).toSet === Set("Hello world", "input text"))
  }
}
