package io.flatmap.ml.regression

import breeze.linalg.DenseVector
import breeze.plot._
import java.awt.Color

case class PlotConfig(
  name: String,
  title: Option[String] = None,
  xlabel: Option[String] = None,
  ylabel: Option[String] = None,
  size: Double = 0.1,
  color: Color = Color.GRAY,
  legend: Boolean = true
)

object Plot {

  def data(X: DenseVector[Double], y: DenseVector[Double])(implicit figure: Figure, cfg: PlotConfig): Unit =
    draw(scatter(X, y, {(_:Int) => cfg.size}, {(_:Int) => cfg.color}, name = cfg.name))

  def error(errors: J)(implicit figure: Figure, cfg: PlotConfig): Unit =
    draw(plot(Array.tabulate(errors.length){_.toDouble}, errors, name = cfg.name))

  def hypothesis(X: Features, theta: Theta)(implicit figure: Figure, cfg: PlotConfig): Unit =
    draw(plot(X(::,1), X * theta, name = cfg.name))

  def draw(series: Series)(implicit figure: Figure, cfg: PlotConfig): Unit = {
    figure.subplot(0) += series
    cfg.ylabel map { figure.subplot(0).ylabel = _ }
    cfg.xlabel map { figure.subplot(0).xlabel = _ }
    cfg.title map { figure.subplot(0).title = _ }
    figure.subplot(0).legend = cfg.legend
    figure.refresh()
  }

}
