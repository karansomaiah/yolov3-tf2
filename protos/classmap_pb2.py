# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/classmap.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/classmap.proto',
  package='yolov3',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x15protos/classmap.proto\x12\x06yolov3\"2\n\nClassLabel\x12\x12\n\nclass_name\x18\x01 \x02(\t\x12\x10\n\x08\x63lass_id\x18\x02 \x02(\x05\"1\n\x08\x43lassMap\x12%\n\tclass_map\x18\x01 \x03(\x0b\x32\x12.yolov3.ClassLabel')
)




_CLASSLABEL = _descriptor.Descriptor(
  name='ClassLabel',
  full_name='yolov3.ClassLabel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='class_name', full_name='yolov3.ClassLabel.class_name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class_id', full_name='yolov3.ClassLabel.class_id', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=33,
  serialized_end=83,
)


_CLASSMAP = _descriptor.Descriptor(
  name='ClassMap',
  full_name='yolov3.ClassMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='class_map', full_name='yolov3.ClassMap.class_map', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=85,
  serialized_end=134,
)

_CLASSMAP.fields_by_name['class_map'].message_type = _CLASSLABEL
DESCRIPTOR.message_types_by_name['ClassLabel'] = _CLASSLABEL
DESCRIPTOR.message_types_by_name['ClassMap'] = _CLASSMAP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ClassLabel = _reflection.GeneratedProtocolMessageType('ClassLabel', (_message.Message,), dict(
  DESCRIPTOR = _CLASSLABEL,
  __module__ = 'protos.classmap_pb2'
  # @@protoc_insertion_point(class_scope:yolov3.ClassLabel)
  ))
_sym_db.RegisterMessage(ClassLabel)

ClassMap = _reflection.GeneratedProtocolMessageType('ClassMap', (_message.Message,), dict(
  DESCRIPTOR = _CLASSMAP,
  __module__ = 'protos.classmap_pb2'
  # @@protoc_insertion_point(class_scope:yolov3.ClassMap)
  ))
_sym_db.RegisterMessage(ClassMap)


# @@protoc_insertion_point(module_scope)