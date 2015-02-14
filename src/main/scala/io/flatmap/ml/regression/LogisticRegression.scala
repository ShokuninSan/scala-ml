package io.flatmap.ml.regression

import breeze.linalg._
import breeze.numerics.log
import breeze.optimize._
import breeze.plot.Figure
import scalaz._
import Scalaz._
import java.awt.Color

object LogisticRegression extends RegressionModel with Optimization {

  def sigmoid(z: Double): Double = 1.0 / (1.0 + Math.pow(scala.math.E, -1 * z))

  override def h(theta: Theta): (Features) => Prediction = (X: Features) => (X * theta) map { sigmoid }

  override def computeCost(X: Features, y: Labels, theta: Theta): Double =
    (1.0/y.length) * sum((-y :* log(h(theta)(X))) :- ((1.0 :- y) :* log(1.0 :- h(theta)(X))))

  def costFunction(X: Features, y: Labels, theta: Theta): (Cost, Gradients) = {
    val cost = computeCost(X, y, theta)
    val gradients = Array.tabulate[Double](theta.length)(i => (1.0/y.length) * sum((h(theta)(X) :- y) :* X(::, i)))
    (cost, DenseVector[Double](gradients))
  }

  def logisticRegression: Unit = {
    implicit val model = Figure("Logistic Regression")
    val data = csvread(new java.io.File("resources/ex2data1.txt"))
    val y = data(::, 2)
    val x = data(::, 0 to 1)
    val m = y.length
    val pos = coordinates(x, y, _ == 1.0)
    val neg = coordinates(x, y, _ == 0.0)
    implicit val modelPlotConfig = PlotConfig(name = "Admitted", title = "Exam scores".some, xlabel = "Exam 1 score".some, ylabel = "Exam 2 score".some, size = 1.0, color = Color.GREEN, legend = false)
    Plot.data(pos._1, pos._2)(model, modelPlotConfig)
    Plot.data(neg._1, neg._2)(model, modelPlotConfig.copy(name = "Not admitted", color = Color.RED))
    val X = DenseMatrix.horzcat(DenseMatrix.ones[Double](m,1), x)
    val theta = DenseVector.zeros[Double](3)
    println(computeCost(X, y, theta))
    val costFn = (theta: Theta) => costFunction(X, y, theta)
    val newTheta = fminunc(costFn, theta)
    println(s"New theta found by fminunc (LBFGS): $newTheta")
    Plot.decisionBoundary(X, y, newTheta)
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
