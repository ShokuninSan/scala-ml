package io.flatmap.ml.regression

import breeze.linalg._
import breeze.plot._

object LinearRegression extends RegressionModel with Optimization {

  private final val data = csvread(new java.io.File("resources/ex1data1.txt"))

  override def h(theta: Theta): Features => Prediction = (X: Features) => X * theta

  override def computeCost(X: Features, y: Labels, theta: Theta): Double = {
    val m = y.length  // # of samples
    val mse = sum((h(theta)(X) - y) :^ 2.0)
    (1.0/(2.0*m)) * mse
  }

  def main(args: Array[String]): Unit = {
    val x = data(::,0)
    val y = data(::,1)
    val m = y.length
    val X = DenseMatrix(DenseVector.ones[Double](m).toArray, data(::,0).toArray).t
    val theta = DenseVector.zeros[Double](2)
    Plot.data(x, y) // scatter plot training data
    gradientDescent(X, y, theta, 0.01, 1500) match {
      case (theta: Theta, history: J) =>
        Plot.hypothesis(X, theta)
        Plot.error(history)
    }
  }

}
