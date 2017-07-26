package org.platanios.tensorflow.api.tensors

import java.nio.ByteBuffer

import org.platanios.tensorflow.api.types.DataType

abstract class RawTensor[+T <: DataType] extends TensorLike {
  override val dataType: T
  def buffer: ByteBuffer
  val order: Order
}
