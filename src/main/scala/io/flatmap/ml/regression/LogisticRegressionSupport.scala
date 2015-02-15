package io.flatmap.ml.regression

import breeze.linalg._
import breeze.plot.Figure
import scalaz._
import Scalaz._
import java.awt.Color

trait LogisticRegressionSupport {

  def initializeModel(title: String, resource: java.io.File, bubbleSize: Double = 1.0, normalizeFn: Double => Double = d => d): (Features, Labels, Theta, Int, Figure, PlotConfig) = {
    implicit val model = Figure(title)
    implicit val modelPlotConfig = PlotConfig(
      name = "Admitted",
      title = "Exam scores".some,
      xlabel = "Exam 1 score".some,
      ylabel = "Exam 2 score".some,
      size = bubbleSize,
      color = Color.GREEN,
      legend = false
    )
    val data = csvread(resource)
    val y = data(::, 2)
    val x = data(::, 0 to 1).mapValues(normalizeFn)
    val m = y.length
    val theta = DenseVector.zeros[Double](3)
    val pos = coordinates(x, y, _ == 1.0)
    val neg = coordinates(x, y, _ == 0.0)
    Plot.data(pos._1, pos._2)
    Plot.data(neg._1, neg._2)(model, modelPlotConfig.copy(name = "Not admitted", color = Color.RED))
    val X = DenseMatrix.horzcat(DenseMatrix.ones[Double](m,1), x)
    (X, y, theta, m, model, modelPlotConfig)
  }

  def coordinates(X: Features, y: Labels, f: Double => Boolean): (DenseVector[Double], DenseVector[Double]) = {
    val i = y.mapPairs((i, e) => (i,e)).findAll { case (i,e) => f(e) }
    val xs = X(i, 0 to 0).toDenseMatrix.toDenseVector
    val ys = X(i, 1 to 1).toDenseMatrix.toDenseVector
    (xs, ys)
  }

}
