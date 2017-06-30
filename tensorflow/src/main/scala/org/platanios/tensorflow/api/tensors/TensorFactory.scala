package org.platanios.tensorflow.api.tensors

import java.nio.charset.Charset
import java.nio.{ByteBuffer, ByteOrder}

import org.platanios.tensorflow.api.DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.tensors.Tensor.allocate
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}


/**
  * @author Sören Brunk
  *
  * @tparam T
  */
trait TensorFactory[T <: DataType] {
  def dataType: T

  def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): Tensor[T]

  def fill(shape: Shape, value: T#ScalaType): Tensor[T]

  def fromTensors(newShape: Shape, tensors: Seq[Tensor[T]], order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): Tensor[T]

  def fromSeq(values: T#ScalaType*): Tensor[T]
  }

trait FixedSizeTensorFactory[T <: FixedSizeDataType] extends TensorFactory[T] {

  override def fromTensors(newShape: Shape, tensors: Seq[Tensor[T]], order: Order): Tensor[T] = {
    val buffer = Tensor.allocate(dataType, newShape, order)
    val newTensor = this.fromBuffer(newShape, buffer, order)
    val newTensorIndexIterator = newTensor.flattenedIndexIterator
    tensors.foreach(t => t.flattenedIndexIterator.foreach(index => {
      newTensor.setElementAtFlattenedIndex(
        newTensorIndexIterator.next(), t.getElementAtFlattenedIndex(index))
    }))
    newTensor
  }

  def fromSeq(values: T#ScalaType*): Tensor[T] = {
    val shape = if (values.length > 1) Shape(values.length) else Shape()
    val buffer = Tensor.allocate[T](dataType, shape, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    val tensor = this.fromBuffer(shape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    val tensorIndexIterator = tensor.flattenedIndexIterator
    values.foreach(value => tensor.setElementAtFlattenedIndex(tensorIndexIterator.next(), value))
    tensor
  }

  override def fill(shape: Shape, value: T#ScalaType): Tensor[T] = ???
}

object TensorFactory {

  implicit object StringTensorFactory extends TensorFactory[STRING.type] {
    override def dataType: STRING.type = STRING

    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[STRING.type] =
      new STRINGTensor(shape, buffer, order)

    override def fill(shape: Shape, value: String): Tensor[STRING.type] = {
      val numStringBytes = value.toString.getBytes(Charset.forName("UTF-8")).length
      val numEncodedBytes = NativeTensor.getEncodedStringSize(numStringBytes)
      val numBytes = shape.numElements * (INT64.byteSize + numEncodedBytes)
      val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
      //val tensor = new StringTensor(inferredShape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
      val baseOffset = INT64.byteSize * shape.numElements
      var index = 0
      var i = 0
      while (i < shape.numElements) {
        STRING.putElementInBuffer(buffer, baseOffset + index, STRING.cast(value))
        INT64.putElementInBuffer(buffer, i * INT64.byteSize, index.toLong)
        index += numEncodedBytes
        i += 1
      }
      new STRINGTensor(shape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    }

    override def fromSeq(values: String*): Unit = {
      // TODO: !!! Make more efficient.
      val shape = if (values.length > 1) Shape(values.length) else Shape()
      var size = INT64.byteSize * values.length
      var i = 0
      while (i < values.length) {
        size += NativeTensor.getEncodedStringSize(values(i).getBytes(Charset.forName("UTF-8")).length)
        i += 1
      }
      val buffer: ByteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder)
      val baseOffset = INT64.byteSize * shape.numElements
      var byteIndex = 0
      i = 0
      while (i < values.length) {
        val numEncodedBytes = STRING.putElementInBuffer(buffer, baseOffset + byteIndex, values(i))
        INT64.putElementInBuffer(buffer, INT64.byteSize * i, byteIndex.toLong)
        byteIndex += numEncodedBytes
        i += 1
      }
      new STRINGTensor(shape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    }

    override def fromTensors(newShape: Shape, tensors: Seq[Tensor[STRING.type]], order: Order): Tensor[STRING.type] = {
      // TODO: Make this more efficient.
      // val numElements = newShape.numElements.get
      var size = 0
      var t = 0
      while (t < tensors.length) {
        size += tensors(t).buffer.capacity() // TODO: This will not work with slices.
        t += 1
      }
      val buffer: ByteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder)
      //val tensor = new StringTensor(newShape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
      val baseOffset = INT64.byteSize * newShape.numElements
      var byteIndex = 0
      var elementIndex = 0
      t = 0
      while (t < tensors.length) {
        val otherBaseOffset = tensors(t).numElements * INT64.byteSize
        var i = 0
        while (i < tensors(t).numElements) {
          val otherOffset = otherBaseOffset +
            INT64.getElementFromBuffer(tensors(t).buffer, i * INT64.byteSize).toInt
          val string = STRING.getElementFromBuffer(tensors(t).buffer, otherOffset)
          val numEncodedBytes = STRING.putElementInBuffer(buffer, baseOffset + byteIndex, string)
          INT64.putElementInBuffer(buffer, elementIndex * INT64.byteSize, byteIndex.toLong)
          byteIndex += numEncodedBytes
          elementIndex += 1
          i += 1
        }
        t += 1
      }
      this.fromBuffer(newShape, buffer, order)
    }
  }

  implicit object BOOLEANTensorFactory extends FixedSizeTensorFactory[BOOLEAN.type] {
    override def dataType: BOOLEAN.type = BOOLEAN
    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[BOOLEAN.type] =
      new BOOLEANTensor(shape, buffer, order)
  }

  implicit object FLOAT32TensorFactory extends FixedSizeTensorFactory[FLOAT32.type] {
    override def dataType: FLOAT32.type = FLOAT32
    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[FLOAT32.type] =
      new FLOAT32Tensor(shape, buffer, order)
  }

  implicit object FLOAT64TensorFactory extends FixedSizeTensorFactory[FLOAT64.type] {
    override def dataType: FLOAT64.type = FLOAT64
    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[FLOAT64.type] =
      new FLOAT64Tensor(shape, buffer, order)
  }

  implicit object INT8TensorFactory extends FixedSizeTensorFactory[INT8.type] {
    override def dataType: INT8.type = INT8
    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[INT8.type] =
      new INT8Tensor(shape, buffer, order)
  }

  implicit object INT16TensorFactory extends FixedSizeTensorFactory[INT16.type] {
    override def dataType: INT16.type = INT16
    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[INT16.type] =
      new INT16Tensor(shape, buffer, order)
  }

  implicit object INT32TensorFactory extends FixedSizeTensorFactory[INT32.type] {
    override def dataType: INT32.type = INT32
    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[INT32.type] =
      new INT32Tensor(shape, buffer, order)
  }

  implicit object INT64TensorFactory extends FixedSizeTensorFactory[INT64.type] {
    override def dataType: INT64.type = INT64
    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[INT64.type] =
      new INT64Tensor(shape, buffer, order)
  }

  implicit object UINT16TensorFactory extends FixedSizeTensorFactory[UINT16.type] {
    override def dataType: UINT16.type = UINT16
    override def fromBuffer(shape: Shape, buffer: ByteBuffer, order: Order): Tensor[UINT16.type] =
      new UINT16Tensor(shape, buffer, order)
  }

}