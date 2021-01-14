# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/eval.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/eval.proto',
  package='yolov3',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x11protos/eval.proto\x12\x06yolov3\"O\n\x04\x45val\x12\x0f\n\x07\x64\x61taset\x18\x01 \x02(\t\x12\x12\n\nbatch_size\x18\x02 \x02(\x05\x12\"\n\x16num_examples_visualize\x18\x03 \x01(\x05:\x02\x31\x30\"/\n\nEvalConfig\x12!\n\x0b\x65val_config\x18\x01 \x02(\x0b\x32\x0c.yolov3.Eval')
)




_EVAL = _descriptor.Descriptor(
  name='Eval',
  full_name='yolov3.Eval',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dataset', full_name='yolov3.Eval.dataset', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='yolov3.Eval.batch_size', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_examples_visualize', full_name='yolov3.Eval.num_examples_visualize', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=10,
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
  serialized_start=29,
  serialized_end=108,
)


_EVALCONFIG = _descriptor.Descriptor(
  name='EvalConfig',
  full_name='yolov3.EvalConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='eval_config', full_name='yolov3.EvalConfig.eval_config', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
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
  serialized_start=110,
  serialized_end=157,
)

_EVALCONFIG.fields_by_name['eval_config'].message_type = _EVAL
DESCRIPTOR.message_types_by_name['Eval'] = _EVAL
DESCRIPTOR.message_types_by_name['EvalConfig'] = _EVALCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Eval = _reflection.GeneratedProtocolMessageType('Eval', (_message.Message,), dict(
  DESCRIPTOR = _EVAL,
  __module__ = 'protos.eval_pb2'
  # @@protoc_insertion_point(class_scope:yolov3.Eval)
  ))
_sym_db.RegisterMessage(Eval)

EvalConfig = _reflection.GeneratedProtocolMessageType('EvalConfig', (_message.Message,), dict(
  DESCRIPTOR = _EVALCONFIG,
  __module__ = 'protos.eval_pb2'
  # @@protoc_insertion_point(class_scope:yolov3.EvalConfig)
  ))
_sym_db.RegisterMessage(EvalConfig)


# @@protoc_insertion_point(module_scope)