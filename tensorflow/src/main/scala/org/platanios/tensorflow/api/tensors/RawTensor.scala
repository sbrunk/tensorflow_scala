package org.platanios.tensorflow.api.tensors

import java.nio.ByteBuffer

import org.platanios.tensorflow.api.types.{DataType, SupportedType}

abstract class RawTensor extends TensorLike {
  override val dataType: DataType
  def buffer: ByteBuffer
  val order: Order
}
