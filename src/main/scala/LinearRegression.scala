import breeze.linalg._
import breeze.plot._

object LinearRegression {

  def plotData(X: DenseVector[Double], y: DenseVector[Double]): Figure = {
    val f = Figure()
    f.subplot(0) +=  scatter(X, y, {(_:Int) => 0.1})
    f.subplot(0).ylabel = "Profit in $10,000s"
    f.subplot(0).xlabel = "Population of City in 10,000s"
    f.subplot(0).title = "Ex1 Data"
    f.refresh()
    f
  }

  /**
   * Hypothesis of all samples of X in one shot.
   *
   * @param X The feature matrix
   * @param theta The parameters
   * @return A vector with prediction results
   */
  def hypothesis(theta: DenseVector[Double]): DenseMatrix[Double] => DenseVector[Double] =
    (X: DenseMatrix[Double]) => X * theta

  def computeCost(X: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): Double = {
    val m = y.length  // # of samples
    val h = hypothesis(theta)
    val mse = sum((h(X) - y) :^ 2.0)
    (1.0/(2.0*m)) * mse
  }

  def gradientDescent(X: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double], alpha: Double, iterations: Int): (DenseVector[Double], Seq[Double]) = {
    val J_history = DenseVector.zeros[Double](iterations).toArray
    var _theta = theta
    val m = y.length
    (0 to iterations - 1) foreach { it =>
      val delta = hypothesis(_theta)(X) - y
      def derivative(featureIndex: Int): Double = (1.0/m) * sum(X(::, featureIndex) :* delta)
      _theta = DenseVector(_theta.toArray.zipWithIndex.map(e => e._1 - alpha * derivative(e._2)))
      J_history(it) = computeCost(X, y, _theta)
    }
    (_theta, J_history)
  }

  def main(args: Array[String]): Unit = {
    val data = csvread(new java.io.File("resources/ex1data1.txt"))
    val x = data(::,0)
    val y = data(::,1)
    val m = y.length
    val X = DenseMatrix(DenseVector.ones[Double](m).toArray, data(::,0).toArray).t
    val theta = DenseVector.zeros[Double](2)
    val f = plotData(x, y) // scatter plot training data
    gradientDescent(X, y, theta, 0.01, 1500) match {
      case tpl =>
        println(s"theta: ${tpl._1}")
        f.subplot(0) += plot(X(::,1), X * tpl._1)
        f.refresh // plot linear hypothesis function
    }
  }

}
