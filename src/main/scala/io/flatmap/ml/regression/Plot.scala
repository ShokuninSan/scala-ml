package io.flatmap.ml.regression

import breeze.linalg.DenseVector
import breeze.plot._

object Plot {

  private final val model = Figure("Linear Regression Model")
  private final val gd = Figure("Gradient Descent")

  def data(name: String, X: DenseVector[Double], y: DenseVector[Double]): Unit = {
    model.subplot(0) +=  scatter(X, y, {(_:Int) => 0.1}, name = name)
    model.subplot(0).ylabel = "Profit in $10,000s"
    model.subplot(0).xlabel = "Population of City in 10,000s"
    model.subplot(0).title = "Linear Regression Model"
    model.refresh()
  }

  def hypothesis(name: String, X: Features, theta: Theta): Unit = {
    model.subplot(0) += plot(X(::,1), X * theta, name = name)
    model.subplot(0).legend = true
    model.refresh()
  }

  def error(errors: J): Unit = {
    gd.subplot(0) += plot(Array.tabulate(errors.length){_.toDouble}, errors)
    gd.subplot(0).ylabel = "J(θ)"
    gd.subplot(0).xlabel = "Iterations"
    gd.subplot(0).title = "Gradient Descent"
    gd.refresh()
  }

}
