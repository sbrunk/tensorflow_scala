/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.{DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER, tensors}
import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.{Index, Indexer, Shape}
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.ops.{Basic, Output, OutputConvertible}
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}
import spire.math.UShort
import java.nio._
import java.nio.charset.Charset


// TODO: Specialized slices (e.g., contiguous).
// TODO: Is there a need to complicate the flattened index function for the plain tensor?
// TODO: Add casting support.
// TODO: Should we keep assuming that tensor shapes are fully defined here?

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Tensor[T <: DataType] private[tensors] (
  val dataType: T,
  val shape: Shape,
  val buffer: ByteBuffer,
  val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)(private[api] implicit val factory: TensorFactory[T])
  extends RawTensor[T] with OutputConvertible {

  private[api] def flattenedIndex(indices: Array[Int]): Int = order.index(shape.asArray, indices)
  private[api] def flattenedIndexIterator: Iterator[Int] = order.indexIterator(shape.asArray)

  private[api] def setElementAtFlattenedIndex(index: Int, value: T#ScalaType): this.type
  private[api] def getElementAtFlattenedIndex(index: Int): T#ScalaType

  def fill(value: T#ScalaType): this.type = {
    for (index <- flattenedIndexIterator)
      setElementAtFlattenedIndex(index, value)
    this
  }

  private[tensors] def newTensor(shape: Shape): Tensor[T] =
    factory.fromBuffer(shape, Tensor.allocate(dataType, shape, order), order)

  def reshape(shape: Shape, copyData: Boolean = true): Tensor[T] = {
    val newShape = this.shape.reshape(shape)
    if (copyData)
      factory(newShape, Tensor.copyBuffer(dataType, newShape, buffer, copy = true, order), order)
    else
      factory(newShape, buffer, order)
  }

  def entriesIterator: Iterator[T#ScalaType] = flattenedIndexIterator.map(getElementAtFlattenedIndex)

  // def update(indices: Array[Int], value: DataType.SupportedScalaType): Unit = {
  //   require(indices.length == rank, "Incomplete set of indices provided.")
  //   val index = order.index(shape.asArray, beginOffsets, strides, indices)
  //   dataType.putElementInBuffer(buffer = buffer, index = index, element = dataType.cast(value))
  // }

  // TODO: Need to improve the syntax here (maybe using implicit conversion to indexer sequences).
  // TODO support casting from other types like in the untyped tensor?
  def update(indexers: Seq[Indexer], tensor: Tensor[T]): Unit = {
    val decoded = Indexer.decode(shape, indexers)
    val sliceShape = Shape.fromSeq(decoded._2)
    if (sliceShape != tensor.shape)
      throw ShapeMismatchException(
        s"Tensor slice shape '$sliceShape' does not match assigned tensor shape '${tensor.shape}'.")
    val stridedIndexIterator = order.indexIterator(decoded._1, decoded._3, decoded._4, decoded._5)
    for ((index, stridedIndex) <- tensor.flattenedIndexIterator zip stridedIndexIterator) {
      // TODO: Avoid casting for tensors with the same data type.
      setElementAtFlattenedIndex(stridedIndex, tensor.getElementAtFlattenedIndex(index))
    }
  }

  // def update(index: Int, tensor: Tensor): Unit = update(Seq[Indexer](index), tensor)
  //
  // def update(indexers: Seq[Indexer], tensor: Tensor): Unit = slice(indexers: _*).set(tensor)

  // TODO: Find a way to add this method for performance benefits.
  // def set(value: SupportedScalaType): Tensor = fill(value)

  def set(tensor: Tensor[T]): this.type = {
    if (shape != tensor.shape && tensor.numElements != 1)
      throw ShapeMismatchException(s"Assigned tensor shape '${tensor.shape}' does not match assignee shape '$shape'")
    for ((index, value) <- flattenedIndexIterator zip tensor.entriesIterator)
      setElementAtFlattenedIndex(index, value)
    this
  }

  def scalar: T#ScalaType = {
    if (numElements != 1)
      throw new IllegalStateException(s"Cannot obtain a scalar value from a non-scalar tensor with shape '$shape'.")
    getElementAtFlattenedIndex(flattenedIndex(Array.fill[Int](shape.rank)(0))) // TODO: Fix this.
  }

  // TODO: !!! Make this return the sub-class tensor type instead.
  def apply(indexers: Indexer*): Tensor[T] = {
    slice(indexers: _*)
    //    if (dataType.byteSize == -1)
    //      throw new IllegalStateException("Cannot index a tensor whose elements have unknown byte size.")
    //    // TODO: Add checks for whether the indexers provided are within bounds.
    //    if ((indexers.length == rank || (indexers.length == 1 && rank == 0)) && indexers.forall(_.isInstanceOf[Index])) {
    //      val index = flattenedIndex(indexers.map(_.asInstanceOf[Index].index).toArray)
    //      dataType.getElementFromBuffer(buffer = buffer, index = index * dataType.byteSize)
    //    } else {
    //      throw InvalidIndexerException(
    //        "Only sequences of single indices that match in length the rank of a tensor, are supported for obtaining the " +
    //            "value of a tensor element.")
    //    }
  }

  // TODO: Make more efficient for contiguous slices.
  def slice(indexers: Indexer*): Tensor[T] = {
    if (shape.rank == 0 && indexers.length == 1
        && indexers.head.isInstanceOf[Index] && indexers.head.asInstanceOf[Index].index == 0) {
      this
    } else {
      val decoded = Indexer.decode(shape, indexers)
      val tensor = newTensor(Shape.fromSeq(decoded._2))
      stridedAssign(tensor, decoded._1, decoded._3, decoded._4, decoded._5)
    }
  }
  // TODO: Use this for creating slices: Buffer.slice().position(sliceStart).limit(sliceSize)

  private[tensors] def stridedAssign(
      tensor: Tensor[T], underlyingTensorDimensions: Array[Int], beginOffsets: Array[Int], endOffsets: Array[Int],
      strides: Array[Int]): Tensor[T] = {
    val stridedIndexIterator = order.indexIterator(underlyingTensorDimensions, beginOffsets, endOffsets, strides)
    for ((newIndex, stridedIndex) <- tensor.flattenedIndexIterator zip stridedIndexIterator)
      tensor.setElementAtFlattenedIndex(newIndex, getElementAtFlattenedIndex(stridedIndex))
    tensor
  }

  override def summarize(maxEntries: Int = numElements): String = {
    // TODO: Fix this by nesting dimensions.
    s"[${entriesIterator.take(maxEntries).mkString(", ")}${if (maxEntries < numElements) ", ..." else ""}]"
  }

  override def toString: String = s"$dataType[${shape.asArray.mkString(", ")}]"

  override def equals(that: Any): Boolean = that match {
    case that: Tensor[T] =>
      this.shape == that.shape &&
          this.dataType == that.dataType &&
          this.entriesIterator.zip(that.entriesIterator).forall(p => p._1 == p._2)
    case _ => false
  }

  override def hashCode(): Int = {
    val prime = 31
    var result = 1
    result = prime * result + dataType.hashCode
    result = prime * result + shape.hashCode
    flattenedIndexIterator.foreach(index => result = prime * result + getElementAtFlattenedIndex(index).hashCode)
    result
  }

  override def toTensor: Tensor[T] = this
  override def toOutput: Output = Basic.constant(this)
}

object Tensor {
  // TODO: [TENSORS] Add constructor methods for numeric tensors and other specific types of tensors.

  def fromSeq[T <: DataType: TensorFactory, U <: T#ScalaType](values: U*): Tensor[T] = {
    val shape = if (values.length > 1) Shape(values.length) else Shape()
    implicitly[TensorFactory[T]].fromSeq(values :_*)
  }

  def apply[T <: DataType](tensors: Tensor[T]*): Tensor[T] = {
    if (tensors.isEmpty)
      throw new IllegalArgumentException("A data type needs to be provided to construct empty tensors.")
    apply(dataType = tensors.map(_.dataType).maxBy(_.priority), tensors: _*)
  }

  def apply[T <: DataType: TensorFactory](dataType: DataType, tensors: Tensor[T]*): Tensor[T] = {
    // TODO: What about column-major string tensors?
    val shape = if (tensors.nonEmpty) tensors.head.shape else Shape()
    if (tensors.nonEmpty)
      require(tensors.tail.forall(_.shape == shape), "All provided tensor shapes must match.")
    val newShape = if (tensors.nonEmpty) Shape(tensors.length +: shape.asArray: _*) else Shape(0)

    implicitly[TensorFactory[T]].fromTensors(newShape, tensors)
  }

  def fill[T <: DataType : TensorFactory, U <: T#ScalaType](value: U, shape: Shape = Shape()): Tensor[T] = {
    // TODO: Add downcasting warnings.
    shape.assertFullyDefined()
    implicitly[TensorFactory[T]].fill(shape, value)
  }

  // TODO: [TENSOR] Add checks for direct/non-direct byte buffers.

  def fromBuffer[T <: DataType : TensorFactory](
    dataType: T, shape: Shape, buffer: ByteBuffer, copy: Boolean = false,
    order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): Tensor[T] = {
    val bufferCopy = copyBuffer(dataType, shape, buffer, copy, order)
    implicitly[TensorFactory[T]].fromBuffer(shape = shape, buffer = bufferCopy, order)
  }

  private[tensors] def copyBuffer(
      dataType: DataType, shape: Shape, buffer: ByteBuffer, copy: Boolean = false,
      order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): ByteBuffer = {
    shape.assertFullyDefined()
    val limit = dataType.byteSize * shape.numElements
    if (!copy && buffer.isDirect) {
      val bufferDuplicate = buffer.duplicate
      bufferDuplicate.limit(limit)
      bufferDuplicate
    } else {
      val bufferCopy = ByteBuffer.allocateDirect(buffer.capacity)
      val readOnlyBufferCopy = buffer.asReadOnlyBuffer
      bufferCopy.put(readOnlyBufferCopy)
      bufferCopy.position(buffer.position)
      bufferCopy.limit(limit)
      bufferCopy.order(buffer.order)
      bufferCopy
    }
  }

  private[api] def allocate[T <: FixedSizeDataType](
    dataType: T, shape: Shape,
    order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): ByteBuffer = {
    shape.assertFullyDefined()
    val numBytes: Int = dataType.byteSize * shape.numElements
    ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
    //new Tensor[T](dataType = dataType, shape = shape, buffer = buffer, order = order)
  }

  private[api] def fromTFNativeHandle(nativeHandle: Long): Tensor[DataType] = {
    DataType.fromCValue(NativeTensor.dataType(nativeHandle)).tensorFromTFNativeHandle(nativeHandle)
  }

  private[api] trait Implicits {
    implicit def scalaValueToTensor(value: Boolean): Tensor[BOOLEAN.type] = Tensor.fill(value)
    implicit def scalaValueToTensor(value: String): Tensor[STRING.type] = Tensor.fill(value)
    implicit def scalaValueToTensor(value: Float): Tensor[FLOAT32.type] = Tensor.fill(value)
    implicit def scalaValueToTensor(value: Double): Tensor[FLOAT64.type ] = Tensor.fill(value)
    implicit def scalaValueToTensor(value: Byte): Tensor[INT8.type] = Tensor.fill(value)
    implicit def scalaValueToTensor(value: Short): Tensor[INT16.type ] = Tensor.fill(value)
    implicit def scalaValueToTensor(value: Int): Tensor[INT32.type] = Tensor.fill(value)
    implicit def scalaValueToTensor(value: Long): Tensor[INT64.type] = Tensor.fill(value)
    implicit def scalaValueToTensor(value: UShort): Tensor[UINT16.type] = Tensor.fill(value)

    implicit def scalaArrayToTensor(value: Array[Boolean]): Tensor[BOOLEAN.type] = Tensor.fromSeq(value: _*)
    // implicit def scalaArrayToTensor(value: Array[String]): Tensor = Tensor.fromSeq(value: _*)(String.supportedType)
    implicit def scalaArrayToTensor(value: Array[Float]): Tensor[FLOAT32.type] = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Double]): Tensor[FLOAT64.type] = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Byte]): Tensor[INT8.type] = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Short]): Tensor[INT16.type] = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Int]): Tensor[INT32.type] = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Long]): Tensor[INT64.type] = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[UShort]): Tensor[UINT16.type] = Tensor.fromSeq(value: _*)

    //implicit def tensorToNumeric(tensor: Tensor): NumericTensor = tensor.asNumeric
    //implicit def tensorToRealNumeric(tensor: Tensor): RealNumericTensor = tensor.asRealNumeric
  }

  private[api] object Implicits extends Implicits

  //  def apply[T: DataType.SupportedScalaTypes#Member](values: T*): Tensor = {
  //    val valueDataType: DataType = DataType.dataTypeOf(values.head)
  //    val shape: Shape = Shape(values.length)
  //    if (valueDataType != DataType.String) {
  //      null
  //    } else {
  //      // TODO: Support String tensors.
  //      throw new UnsupportedOperationException(
  //        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
  //            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
  //    }
  //  }
  //
  //  /** Creates a [[Tensor]].
  //    *
  //    * The resulting tensor is populated with values of type `dataType`, as specified by the arguments `value` and
  //    * (optionally) `shape` (see examples below).
  //    *
  //    * The argument `value` can be a constant value, or an array (potentially multi-dimensional) with elements of type
  //    * `dataType`. If `value` is a one-dimensional array, then its length should be less than or equal to the number of
  //    * elements implied by the `shape` argument (if specified). In the case where the array length is less than the
  //    * number of elements specified by `shape`, the last element in the array will be used to fill the remaining entries.
  //    *
  //    * The argument `dataType` is optional. If not specified, then its value is inferred from the type of `value`.
  //    *
  //    * The argument `shape` is optional. If present, it specifies the dimensions of the resulting tensor. If not present,
  //    * the shape of `value` is used.
  //    *
  //    * **IMPORTANT NOTE** The data type argument and the shape arguments are not currently being used.
  //    *
  //    * @param  value       A constant value of data type `dataType`.
  //    * @param  dataType    Data type of the resulting tensor. If not provided, its value will be inferred from the type
  //    *                     of `value`.
  //    * @param  shape       Shape of the resulting tensor.
  //    * @param  verifyShape If `true` and `shape` is not `null`, then the shape of `value` will be verified (i.e., checked
  //    *                     to see if it is equal to the provided shape.
  //    * @return Created tensor.
  //    * @throws InvalidShapeException If `shape != null`, `verifyShape == true`, and the shape of values does not match
  //    *                               the provided `shape`.
  //    */
  //  def create(value: Any, dataType: DataType = null, shape: Shape = null, verifyShape: Boolean = false): Tensor = {
  //    val valueDataType: DataType = DataType.dataTypeOf(value)
  //    val inferredDataType: DataType = if (dataType == null) valueDataType else dataType
  //    val inferredShape: Shape = if (shape == null) Tensor.shape(value) else shape
  //    // TODO: !!! Fix this so that it actually does verify the shape and the data type and does appropriate type casts.
  //    if (inferredDataType != DataType.String) {
  //      val numElements = inferredShape.numElements.get
  //      val byteSize = inferredDataType.byteSize * numElements
  //      val nativeHandle = NativeTensor.allocate(inferredDataType.cValue, inferredShape.asArray, byteSize)
  //      if (inferredDataType != valueDataType) {
  //        val tensor: Tensor = allocateForBuffer(dataType, inferredShape, numElements)
  //        castAndWriteTo(tensor.buffer, value, dataType)
  //        tensor
  //      } else {
  //        NativeTensor.setValue(nativeHandle, value)
  //        Tensor(dataType = inferredDataType, shape = inferredShape, nativeHandle = nativeHandle)
  //      }
  //    } else if (inferredShape.rank != 0) {
  //      // TODO: Support String tensors.
  //      throw new UnsupportedOperationException(
  //        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
  //            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
  //    } else {
  //      val nativeHandle = NativeTensor.allocateScalarBytes(value.asInstanceOf[Array[Byte]])
  //      Tensor(dataType = inferredDataType, shape = inferredShape, nativeHandle = nativeHandle)
  //    }
  //  }
  //
  //  private[this] def castAndWriteTo(buffer: ByteBuffer, value: Any, dataType: DataType): Unit = {
  //    // TODO: May be doable more efficiently.
  //    def writeToHelper(buffer: ByteBuffer, bufferIndex: Int, value: Any, dataType: DataType): Int = {
  //      value match {
  //        case array: Array[_] =>
  //          var bytesWritten = 0
  //          var i = 0
  //          while (i < array.length) {
  //            bytesWritten += writeToHelper(buffer, bufferIndex + bytesWritten, array(i), dataType)
  //            i += 1
  //          }
  //          bytesWritten
  //        case scalar =>
  //          dataType.putElementInBuffer(buffer, bufferIndex, dataType.cast(scalar))
  //          dataType.byteSize
  //      }
  //    }
  //    writeToHelper(buffer, 0, value, dataType)
  //  }
  //
  //  def create(shape: Shape, data: FloatBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Float32, shape, data.remaining())
  //    tensor.buffer.asFloatBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: DoubleBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Float64, shape, data.remaining())
  //    tensor.buffer.asDoubleBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: IntBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Int32, shape, data.remaining())
  //    tensor.buffer.asIntBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: LongBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Int64, shape, data.remaining())
  //    tensor.buffer.asLongBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(dataType: DataType, shape: Shape, data: ByteBuffer): Tensor = {
  //    val numRemaining: Int = {
  //      if (dataType != DataType.String) {
  //        if (data.remaining() % dataType.byteSize != 0)
  //          throw new IllegalArgumentException(s"A byte buffer with ${data.remaining()} bytes is not compatible with a " +
  //                                                 s"${dataType.toString} Tensor (${dataType.byteSize} bytes/element).")
  //        data.remaining() / dataType.byteSize
  //      } else {
  //        data.remaining()
  //      }
  //    }
  //    val tensor: Tensor = allocateForBuffer(dataType, shape, numRemaining)
  //    tensor.buffer.put(data)
  //    tensor
  //  }
  //  // Helper function to allocate a Tensor for the create() methods that create a Tensor from
  //  // a java.nio.Buffer.
  //  private def allocateForBuffer(dataType: DataType, shape: Shape, numBuffered: Int): Tensor = {
  //    val size: Long = shape.numElements.get
  //    val numBytes: Long = {
  //      if (dataType != DataType.String) {
  //        if (numBuffered != size)
  //          throw incompatibleBufferException(numBuffered, shape)
  //        size * dataType.byteSize
  //      } else {
  //        // DataType.String tensor encoded in a ByteBuffer.
  //        numBuffered
  //      }
  //    }
  //    val nativeHandle: Long = NativeTensor.allocate(dataType.cValue, shape.asArray.clone(), numBytes)
  //    Tensor(dataType = dataType, shape = shape, nativeHandle = nativeHandle)
  //  }
  //
  //  private def incompatibleBufferException(buffer: Buffer, dataType: DataType): IllegalArgumentException = {
  //    new IllegalArgumentException(s"Cannot use ${buffer.getClass.getName} with a Tensor of type $dataType.")
  //  }
  //
  //  private def incompatibleBufferException(numElements: Int, shape: Shape): IllegalArgumentException = {
  //    new IllegalArgumentException(
  //      s"A buffer with $numElements elements is not compatible with a Tensor with shape '$shape'.")
  //  }
  //
  //  private def rank(value: Any): Int = {
  //    value match {
  //      // Array[Byte] is a DataType.STRING scalar.
  //      case _: Array[Byte] => 0
  //      case value: Array[_] => 1 + rank(value(0))
  //      case _ => 0
  //    }
  //  }
  //
  //  private def shape(value: Any): Shape = {
  //    def fillShape(value: Any, axis: Int, shape: Array[Long]): Unit = {
  //      if (shape != null && axis != shape.length) {
  //        if (shape(axis) == 0) {
  //          value match {
  //            case value: Array[_] => shape(axis) = value.length
  //            case _ => shape(axis) = 1
  //          }
  //        } else {
  //          val mismatchedShape = value match {
  //            case value: Array[_] => (shape(axis) != value.length, value.length)
  //            case _ => (shape(axis) != 1, 1)
  //          }
  //          if (mismatchedShape._1)
  //            throw new IllegalArgumentException(
  //              s"Mismatched lengths (${shape(axis)} and ${mismatchedShape._2}) for dimension $axis.")
  //        }
  //        value match {
  //          case value: Array[_] =>
  //            var i = 0
  //            while (i < value.length) {
  //              fillShape(value(i), axis + 1, shape)
  //              i += 1
  //            }
  //        }
  //      }
  //    }
  //
  //    val shapeArray = Array.ofDim[Long](rank(value))
  //    fillShape(value = value, axis = 0, shape = shapeArray)
  //    Shape.fromSeq(shapeArray)
  //  }
}
