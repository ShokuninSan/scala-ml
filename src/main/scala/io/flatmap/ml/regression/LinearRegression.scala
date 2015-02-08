package io.flatmap.ml.regression

import breeze.linalg._

object LinearRegression extends RegressionModel with Optimization {

  lazy val data = csvread(new java.io.File("resources/ex1data1.txt"))
  lazy val x = data(::,0)
  lazy val y = data(::,1)
  lazy val m = y.length
  lazy val X = DenseMatrix(DenseVector.ones[Double](m).toArray, data(::,0).toArray).t
  lazy val theta = DenseVector.zeros[Double](2)

  override def h(theta: Theta): Features => Prediction = (X: Features) => X * theta

  override def computeCost(X: Features, y: Labels, theta: Theta): Double = {
    val mse = sum((h(theta)(X) - y) :^ 2.0)
    (1.0/(2.0*m)) * mse
  }

  def main(args: Array[String]): Unit = {
    Plot.data("Profit/Population", x, y) // scatter plot training data
    gradientDescent(X, y, theta, alpha = 0.01, iterations = 1500) match {
      case (theta: Theta, history: J) =>
        Plot.hypothesis("Gradient Descent", X, theta)
        Plot.error(history)
    }
    normalEquation(X, y) match {
      case theta => Plot.hypothesis("Normal Equation", X, theta)
    }
  }

}
