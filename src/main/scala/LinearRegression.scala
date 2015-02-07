import breeze.linalg._
import breeze.plot._

object LinearRegression {

  type Theta = DenseVector[Double]
  type J = Array[Double]
  type Features = DenseMatrix[Double]
  type Labels = DenseVector[Double]
  type Prediction = DenseVector[Double]

  def plotData(X: DenseVector[Double], y: DenseVector[Double]): Figure = {
    val f = Figure()
    f.subplot(0) +=  scatter(X, y, {(_:Int) => 0.1})
    f.subplot(0).ylabel = "Profit in $10,000s"
    f.subplot(0).xlabel = "Population of City in 10,000s"
    f.subplot(0).title = "Ex1 Data"
    f.refresh()
    f
  }

  def h(theta: Theta): Features => Prediction = (X: Features) => X * theta

  def computeCost(X: Features, y: Labels, theta: Theta): Double = {
    val m = y.length  // # of samples
    val mse = sum((h(theta)(X) - y) :^ 2.0)
    (1.0/(2.0*m)) * mse
  }

  private def updateTheta(theta: Theta, f: ((Double, Int)) => Double): Theta = DenseVector(theta.toArray.zipWithIndex.map(f))

  def gradientDescent(X: Features, y: Labels, theta: Theta, alpha: Double, iterations: Int): (Theta, J) = {
    def derivative(theta: Theta, featureIndex: Int) = (1.0/y.length) * sum((h(theta)(X) - y) :* X(::, featureIndex))
    val J_history = DenseVector.zeros[Double](iterations).toArray
    (0 to iterations - 1).foldLeft ((theta, J_history)) { (g, i) =>
      val _theta = updateTheta(g._1, e => e._1 - alpha * derivative(g._1, e._2))
      g._2(i) = computeCost(X, y, _theta)
      (_theta, g._2)
    }
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
