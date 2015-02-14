package io.flatmap.ml.regression

import breeze.linalg.DenseVector
import breeze.plot._

case class PlotConfig(name: String, title: Option[String] = None, xlabel: Option[String] = None, ylabel: Option[String] = None)

object Plot {

  def data(X: DenseVector[Double], y: DenseVector[Double])(implicit figure: Figure, cfg: PlotConfig): Unit =
    draw(scatter(X, y, {(_:Int) => 0.1}, name = cfg.name))

  def error(errors: J)(implicit figure: Figure, cfg: PlotConfig): Unit =
    draw(plot(Array.tabulate(errors.length){_.toDouble}, errors, name = cfg.name))

  def hypothesis(X: Features, theta: Theta)(implicit figure: Figure, cfg: PlotConfig): Unit =
    draw(plot(X(::,1), X * theta, name = cfg.name))

  def draw(series: Series)(implicit figure: Figure, cfg: PlotConfig): Unit = {
    figure.subplot(0) += series
    cfg.ylabel map { figure.subplot(0).ylabel = _ }
    cfg.xlabel map { figure.subplot(0).xlabel = _ }
    cfg.title map { figure.subplot(0).title = _ }
    figure.subplot(0).legend = true
    figure.refresh()
  }

}
