package io.flatmap.ml.regression

import breeze.linalg._
import breeze.stats._

trait Optimization {

  self: RegressionModel =>

  def normalEquation(X: Features, y: Labels): Theta = pinv(X.t * X) * X.t * y

  def gradientDescent(X: Features, y: Labels, theta: Theta, alpha: Double, iterations: Int): (Theta, J) = {
    def derivative(theta: Theta, featureIndex: Int) = (1.0/y.length) * sum((h(theta)(X) - y) :* X(::, featureIndex))
    val J_history = DenseVector.zeros[Double](iterations).toArray
    (0 to iterations - 1).foldLeft ((theta, J_history)) { (g, i) =>
      val _theta = updateTheta(g._1, e => e._1 - alpha * derivative(g._1, e._2))
      g._2(i) = computeCost(X, y, _theta)
      (_theta, g._2)
    }
  }

  def featureNormalize(X: Features): (Features, DenseVector[Double] => DenseVector[Double]) = {
    val mu = mean(X(::,*)).toDenseVector
    val sigma = stddev(X(::,*)).toDenseVector
    val normX = X(*,::) map { v => (v - mu) / sigma }
    (normX, (x: DenseVector[Double]) => (x - mu) / sigma)
  }

  private def updateTheta(theta: Theta, f: ((Double, Int)) => Double): Theta =
    DenseVector(theta.toArray.zipWithIndex.map(f))

}
