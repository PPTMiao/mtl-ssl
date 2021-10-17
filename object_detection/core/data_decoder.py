# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Interface for data decoders.

Data decoders decode the input data and return a dictionary of tensors keyed by
the entries in core.reader.Fields.
"""
from abc import ABCMeta    # abc为抽象基类模块
from abc import abstractmethod
# 抽象类用来描述一种类型应该具备的基本特征与功能，具体如何去完成这些行为由子类通过方法重写来完成。
# 抽象方法指只有功能声明，没有功能主体实现的方法。具有抽象方法的类一定为抽象类。
# 抽象类无法直接创建对象，只能被子类继承后，创建子类对象。
class DataDecoder(object):
  """Interface for data decoders."""
  __metaclass__ = ABCMeta

  @abstractmethod
  def decode(self, data):
    """Return a single image and associated labels.

    Args:
      data: a string tensor holding a serialized protocol buffer corresponding
        to data for a single image.

    Returns:
      tensor_dict: a dictionary containing tensors. Possible keys are defined in
          reader.Fields.
    """
    pass
