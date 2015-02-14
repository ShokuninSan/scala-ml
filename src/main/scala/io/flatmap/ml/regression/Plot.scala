package io.flatmap.ml.regression

import breeze.linalg.DenseVector
import breeze.plot._

case class PlotConfig(name: String, title: String = "", xlabel: String = "", ylabel: String = "")

object Plot {

  def data(X: DenseVector[Double], y: DenseVector[Double])(implicit figure: Figure, cfg: PlotConfig): Unit = {
    figure.subplot(0) +=  scatter(X, y, {(_:Int) => 0.1}, name = cfg.name)
    figure.subplot(0).ylabel = cfg.ylabel
    figure.subplot(0).xlabel = cfg.xlabel
    figure.subplot(0).title = cfg.title
    figure.refresh()
  }

  def hypothesis(X: Features, theta: Theta)(implicit figure: Figure, cfg: PlotConfig): Unit = {
    figure.subplot(0) += plot(X(::,1), X * theta, name = cfg.name)
    figure.subplot(0).legend = true
    figure.refresh()
  }

  def error(errors: J)(implicit figure: Figure, cfg: PlotConfig): Unit = {
    figure.subplot(0) += plot(Array.tabulate(errors.length){_.toDouble}, errors)
    figure.subplot(0).ylabel = cfg.ylabel
    figure.subplot(0).xlabel = cfg.xlabel
    figure.subplot(0).title = cfg.title
    figure.refresh()
  }

}
