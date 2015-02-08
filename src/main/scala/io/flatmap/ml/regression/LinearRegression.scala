package io.flatmap.ml.regression

import breeze.linalg._

object LinearRegression extends RegressionModel with Optimization {

  override def h(theta: Theta): Features => Prediction = (X: Features) => X * theta

  override def computeCost(X: Features, y: Labels, theta: Theta): Double = {
    val mse = sum((h(theta)(X) - y) :^ 2.0)
    (1.0/(2.0*y.length)) * mse
  }

  def univariateLinearRegressionEval: Unit = {
    val data = csvread(new java.io.File("resources/ex1data1.txt"))
    val x = data(::,0)
    val y = data(::,1)
    val m = y.length
    val X = DenseMatrix(DenseVector.ones[Double](m).toArray, data(::,0).toArray).t
    val theta = DenseVector.zeros[Double](2)
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

  def multivariateLinearRegressionEval: Unit = {
    val data = csvread(new java.io.File("resources/ex1data2.txt"))
    val x = data(::,0 to 1)
    val y = data(::,2)
    val m = y.length
    featureNormalize(x) match {
      case (features, normalizingFn) =>
        val X = DenseMatrix.horzcat(DenseMatrix.ones[Double](m,1), features)
        val theta = DenseVector.zeros[Double](3)
        gradientDescent(X, y, theta, 0.09, 50) match {
          case (theta, history) =>
            val features = DenseVector.vertcat(DenseVector(1.0), normalizingFn(DenseVector(1650.0, 3.0)))
            val predictedPrice = sum(features :* theta)
            println(s"Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $predictedPrice")
        }
    }
  }

  def main(args: Array[String]): Unit = {
    univariateLinearRegressionEval
    multivariateLinearRegressionEval
  }

}
