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

import java.nio.ByteBuffer

import org.platanios.tensorflow.api.DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.types._

/**
  * @author Sören Brunk
  */
private[api]  class FLOAT64Tensor(
  override val shape: Shape,
  override val buffer: ByteBuffer,
  override val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
  extends Tensor[FLOAT64.type](FLOAT64, shape, buffer, order) {

  override private[api] def setElementAtFlattenedIndex(index: Int, value: Double): this.type = {
    dataType.putElementInBuffer(buffer, index * dataType.byteSize, value)
    this
  }

  override private[api] def getElementAtFlattenedIndex(index: Int): Double = {
    dataType.getElementFromBuffer(buffer, index * dataType.byteSize)
  }
}