package io.flatmap.ml.regression

import breeze.linalg._
import breeze.numerics.log

object LogisticRegression extends RegressionModel with Optimization with LogisticRegressionSupport {

  override def h(theta: Theta): (Features) => Prediction = (X: Features) => (X * theta) map { sigmoid }

  override def computeCost(X: Features, y: Labels, theta: Theta): Double =
    (1.0/y.length) * sum((-y :* log(h(theta)(X))) :- ((1.0 :- y) :* log(1.0 :- h(theta)(X))))

  def sigmoid(z: Double): Double = 1.0 / (1.0 + Math.pow(scala.math.E, -1 * z))

  def costFunction(X: Features, y: Labels, theta: Theta): (Cost, Gradients) = {
    val cost = computeCost(X, y, theta)
    val gradients = Array.tabulate[Double](theta.length)(i => (1.0/y.length) * sum((h(theta)(X) :- y) :* X(::, i)))
    (cost, DenseVector[Double](gradients))
  }

  def linearDecisionBoundaryEval: Unit = {
    val (x, y, theta, m, figure, plotConfig) =
      initializeModel("Logistic Regression with linear decision boundary", new java.io.File("resources/ex2data1.txt"))
    println("Cost at initial theta (zeros): " + computeCost(x, y, theta))
    val newTheta = fminunc((theta: Theta) => costFunction(x, y, theta), theta)
    println(s"New theta found by fminunc (LBFGS): $newTheta")
    Plot.decisionBoundary(x, y, newTheta)(figure, plotConfig)
    val probability = h(newTheta)(DenseVector[Double](1.0, 45.0, 85.0).toDenseMatrix)
    println(s"A student with an Exam 1 score of 45 and an Exam 2 score of 85 has admission probability: $probability")
  }

  def polynomialDecisionBoudaryEval: Unit = {
    val (x, y, theta, m, figure, plotConfig) =
      initializeModel(
        "Logistic Regression with polynomial decision boundary",
        new java.io.File("resources/ex2data2.txt"),
        bubbleSize = 2.0,
        normalizeFn = d => d*100
      )
    println("Cost at initial theta (zeros): " + computeCost(x, y, theta))
  }

  def main(args: Array[String]): Unit = {
    linearDecisionBoundaryEval
    polynomialDecisionBoudaryEval
  }

}
