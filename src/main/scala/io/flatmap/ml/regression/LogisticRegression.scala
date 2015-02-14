package io.flatmap.ml.regression

import breeze.linalg._
import breeze.plot.Figure
import scalaz._
import Scalaz._
import java.awt.Color

object LogisticRegression extends RegressionModel with Optimization {

  override def h(theta: Theta): (Features) => Prediction = ???

  override def computeCost(X: Features, y: Labels, theta: Theta): Double = ???

  def logisticRegression: Unit = {
    val model = Figure("Logistic Regression")
    val data = csvread(new java.io.File("resources/ex2data1.txt"))
    val y = data(::, 2)
    val X = data(::, 0 to 1)
    val pos = coordinates(X, y, _ == 1.0)
    val neg = coordinates(X, y, _ == 0.0)
    val modelPlotConfig = PlotConfig(name = "Admitted", title = "Exam scores".some, xlabel = "Exam 1 score".some, ylabel = "Exam 2 score".some, size = 1.0, color = Color.GREEN, legend = false)
    Plot.data(pos._1, pos._2)(model, modelPlotConfig)
    Plot.data(neg._1, neg._2)(model, modelPlotConfig.copy(name = "Not admitted", color = Color.RED))
  }

  private def coordinates(X: Features, y: Labels, f: Double => Boolean): (DenseVector[Double], DenseVector[Double]) = {
    val i = y.mapPairs((i, e) => (i,e)).findAll { case (i,e) => f(e) }
    val xs = X(i, 0 to 0).toDenseMatrix.toDenseVector
    val ys = X(i, 1 to 1).toDenseMatrix.toDenseVector
    (xs, ys)
  }

  def main(args: Array[String]): Unit = {
    logisticRegression
  }

}
