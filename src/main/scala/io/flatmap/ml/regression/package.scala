package io.flatmap.ml

import breeze.linalg.{DenseVector, DenseMatrix}

package object regression {

  type Theta = DenseVector[Double]
  type J = Array[Double]
  type Cost = Double
  type Features = DenseMatrix[Double]
  type Labels = DenseVector[Double]
  type Prediction = DenseVector[Double]
  type Mu = DenseVector[Double]
  type Sigma = DenseVector[Double]
  type Gradients = DenseVector[Double]

}
