package io.flatmap.ml.regression

import breeze.linalg._
import breeze.plot.Figure
import scalaz._
import Scalaz._
import java.awt.Color

trait LogisticRegressionSupport {

  def initializeLinearModel: (Features, Labels, Theta, Int, Figure, PlotConfig) = {
    val plotConfig = PlotConfig(
      name = "Admitted",
      title = "Exam scores".some,
      xlabel = "Exam 1 score".some,
      ylabel = "Exam 2 score".some,
      size = 1.0,
      color = Color.GREEN,
      legend = false
    )
    val (x, y, theta, m, figure) = initializeModel(
      "Logistic Regression with linear decision boundary",
      new java.io.File("resources/ex2data1.txt"),
      plotConfig1 = plotConfig,
      plotConfig2 = plotConfig.copy(name = "Not admitted", color = Color.RED)
    )
    (x, y, theta, m, figure, plotConfig)
  }

  def initializePolynomialModel: (Features, Labels, Theta, Int, Figure, PlotConfig) = {
    val plotConfig = PlotConfig(
      name = "Passed",
      title = "Microchip tests".some,
      xlabel = "Microchip test 1".some,
      ylabel = "Microchip test 2".some,
      size = 2.0,
      color = Color.GREEN,
      legend = false
    )
    val (x, y, theta, m, figure) = initializeModel(
      "Logistic Regression with polynomial decision boundary",
      new java.io.File("resources/ex2data2.txt"),
      plotConfig1 = plotConfig,
      plotConfig2 = plotConfig.copy(name = "Failed", color = Color.RED),
      normalizeFn = d => d*100 // Need to increase values for scatter plot to visualize data as expected
    )
    (x, y, theta, m, figure, plotConfig)
  }

  def initializeModel(title: String, resource: java.io.File, plotConfig1: PlotConfig, plotConfig2: PlotConfig, normalizeFn: Double => Double = d => d): (Features, Labels, Theta, Int, Figure) = {
    implicit val model = Figure(title)
    implicit val modelPlotConfig = plotConfig1
    val data = csvread(resource)
    val y = data(::, 2)
    val x = data(::, 0 to 1).mapValues(normalizeFn)
    val m = y.length
    val theta = DenseVector.zeros[Double](3)
    val pos = coordinates(x, y, _ == 1.0)
    val neg = coordinates(x, y, _ == 0.0)
    Plot.data(pos._1, pos._2)
    Plot.data(neg._1, neg._2)(model, plotConfig2)
    val X = DenseMatrix.horzcat(DenseMatrix.ones[Double](m,1), x)
    (X, y, theta, m, model)
  }

  def coordinates(X: Features, y: Labels, f: Double => Boolean): (DenseVector[Double], DenseVector[Double]) = {
    val i = y.mapPairs((i, e) => (i,e)).findAll { case (i,e) => f(e) }
    val xs = X(i, 0 to 0).toDenseMatrix.toDenseVector
    val ys = X(i, 1 to 1).toDenseMatrix.toDenseVector
    (xs, ys)
  }

  /**
   *  Feature mapping function to polynomial features
   *
   *  mapFeatures(X1, X2) maps the two input features to quadratic features used in the regularization exercise.
   *  Inputs X1, X2 must be the same size.
   *
   * @param x1
   * @param x2
   * @param degree
   * @return A new feature matrix with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
   */
  def mapFeatures(x1: DenseVector[Double], x2: DenseVector[Double], degree: Int = 6): Features = {
    assert(x1.length == x2.length)
    val xs: Seq[DenseVector[Double]] = for {
      i <- 1 to degree
      j <- 0 to i
    } yield (x1 :^ (i-j).toDouble) :* (x2 :^ (j.toDouble))
    val m1 = DenseMatrix.ones[Double](x1.length,1)
    val m2 = DenseMatrix(xs map { _.toArray }: _*).t
    DenseMatrix.horzcat(m1, m2)
  }


}
