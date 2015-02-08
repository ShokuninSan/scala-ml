package io.flatmap.ml

import breeze.linalg.{DenseMatrix, DenseVector}

package object regression {

  type Theta = DenseVector[Double]
  type J = Array[Double]
  type Features = DenseMatrix[Double]
  type Labels = DenseVector[Double]
  type Prediction = DenseVector[Double]
  type Mu = DenseVector[Double]
  type Sigma = DenseVector[Double]

}
