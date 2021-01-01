# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/route.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/route.proto',
  package='yolov3',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x12protos/route.proto\x12\x06yolov3\":\n\x05Route\x12\x0f\n\x07indices\x18\x01 \x03(\x05\x12\n\n\x02id\x18\x02 \x02(\x05\x12\x14\n\x08input_id\x18\x03 \x01(\x05:\x02-1\"0\n\nRouteLayer\x12\"\n\x0broute_layer\x18\x01 \x02(\x0b\x32\r.yolov3.Route')
)




_ROUTE = _descriptor.Descriptor(
  name='Route',
  full_name='yolov3.Route',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='indices', full_name='yolov3.Route.indices', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='yolov3.Route.id', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_id', full_name='yolov3.Route.input_id', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
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
  serialized_start=30,
  serialized_end=88,
)


_ROUTELAYER = _descriptor.Descriptor(
  name='RouteLayer',
  full_name='yolov3.RouteLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='route_layer', full_name='yolov3.RouteLayer.route_layer', index=0,
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
  serialized_start=90,
  serialized_end=138,
)

_ROUTELAYER.fields_by_name['route_layer'].message_type = _ROUTE
DESCRIPTOR.message_types_by_name['Route'] = _ROUTE
DESCRIPTOR.message_types_by_name['RouteLayer'] = _ROUTELAYER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Route = _reflection.GeneratedProtocolMessageType('Route', (_message.Message,), dict(
  DESCRIPTOR = _ROUTE,
  __module__ = 'protos.route_pb2'
  # @@protoc_insertion_point(class_scope:yolov3.Route)
  ))
_sym_db.RegisterMessage(Route)

RouteLayer = _reflection.GeneratedProtocolMessageType('RouteLayer', (_message.Message,), dict(
  DESCRIPTOR = _ROUTELAYER,
  __module__ = 'protos.route_pb2'
  # @@protoc_insertion_point(class_scope:yolov3.RouteLayer)
  ))
_sym_db.RegisterMessage(RouteLayer)


# @@protoc_insertion_point(module_scope)