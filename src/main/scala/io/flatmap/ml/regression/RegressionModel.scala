package io.flatmap.ml.regression

trait RegressionModel {

  def h(theta: Theta): Features => Prediction

  def computeCost(X: Features, y: Labels, theta: Theta): Double

}
