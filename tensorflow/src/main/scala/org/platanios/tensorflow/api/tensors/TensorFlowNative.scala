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

import org.platanios.tensorflow.api.Closeable
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.Disposer
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import java.nio.ByteOrder

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] object TensorFlowNative {
  private[api] class DataTypeOps(val dataType: DataType) extends AnyVal {
    private[api] def tensorFromTFNativeHandle(nativeHandle: Long): Tensor[DataType] = {
      val tensor = dataType match {
        case STRING | _: FixedSizeDataType =>
          new Tensor(
            dataType = dataType, shape = Shape.fromSeq(NativeTensor.shape(nativeHandle).map(_.toInt)),
            buffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder),
            order = RowMajorOrder)
        case d => throw InvalidDataTypeException(s"Tensors with data type '$d' are not supported on the Scala side.")
      }
      // Keep track of references in the Scala side and notify the native library when the tensor is not referenced
      // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
      // potential memory leak.
      Disposer.add(tensor, () => NativeTensor.delete(nativeHandle))
      tensor
    }
  }

  private[api] final class NativeView(private[api] var nativeHandle: Long) extends Closeable {
    override def close(): Unit = {
      if (nativeHandle != 0) {
        NativeTensor.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }

  private[api] class NativeViewOps(tensor: Tensor) {
    private[api] def nativeView: NativeView = {
      if (tensor.order != RowMajorOrder)
        throw new IllegalArgumentException("Only row-major tensors can be used in the TensorFlow native library.")
      new NativeView(NativeTensor.fromBuffer(
        tensor.buffer, tensor.dataType.cValue, tensor.shape.asArray.map(_.toLong), tensor.buffer.capacity()))
    }
  }

  private[api] trait Implicits {
    implicit def dataTypeOps(dataType: DataType): DataTypeOps = new DataTypeOps(dataType)
    implicit def nativeViewOps(tensor: Tensor): NativeViewOps = new NativeViewOps(tensor)
  }

  private[api] object Implicits extends Implicits
}
