import argparse

def main():
  parser = argparse.ArgumentParser(description="SSCMA Node Client Configuration")
  
  # MQTT Broker Configuration
  parser.add_argument("--broker", type=str, default="192.168.88.232", help="MQTT broker address")
  parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
  
  # Camera Node Configuration
  parser.add_argument("--camera_option", type=str, default='2', help="Camera option")
  parser.add_argument("--camera_audio", type=bool, default=True, help="Enable/disable camera audio")
  parser.add_argument("--camera_preview", type=bool, default=False, help="Enable/disable camera preview")
  
  # Model Node Configuration
  parser.add_argument("--model_uri", type=str, default="", help="Model URI")
  parser.add_argument("--model_tscore", type=float, default=0.45, help="Model threshold score")
  parser.add_argument("--model_tiou", type=float, default=0.35, help="Model threshold IoU")
  parser.add_argument("--model_topk", type=int, default=0, help="Model top-k")
  parser.add_argument("--model_labels", type=str, nargs="*", default=[], help="Model labels")
  parser.add_argument("--model_debug", type=bool, default=False, help="Enable/disable model debug mode")
  parser.add_argument("--model_audio", type=bool, default=True, help="Enable/disable model audio")
  parser.add_argument("--model_trace", type=bool, default=False, help="Enable/disable model trace")
  parser.add_argument("--model_counting", type=bool, default=False, help="Enable/disable model counting")
  parser.add_argument("--model_splitter", type=str, nargs="*", default=[], help="Model splitter")
  
  # Streaming Node Configuration
  parser.add_argument("--stream_protocol", type=str, default="rtsp", help="Streaming protocol")
  parser.add_argument("--stream_port", type=int, default=8554, help="Streaming port")
  parser.add_argument("--stream_session", type=str, default="live", help="Streaming session")
  parser.add_argument("--stream_user", type=str, default="admin", help="Streaming user")
  parser.add_argument("--stream_password", type=str, default="admin", help="Streaming password")
  
  # Save Node Configuration
  parser.add_argument("--save_storage", type=str, default="local", help="Save storage type")
  parser.add_argument("--save_duration", type=int, default=-1, help="Save duration")
  parser.add_argument("--save_slice_time", type=int, default=300, help="Save slice time")
  
  args = parser.parse_args()

  # Initialize the SSCMA Node Client
  client = SSCMANodeClient("recamera", "v0")
  
  # Create nodes with the provided configurations
  camera = CameraNode(client, option=args.camera_option, audio=args.camera_audio, preview=args.camera_preview)
  model = ModelNode(client, uri=args.model_uri, tscore=args.model_tscore, tiou=args.model_tiou, topk=args.model_topk,labels=args.model_labels, debug=args.model_debug, audio=args.model_audio, trace=args.model_trace,counting=args.model_counting, splitter=args.model_splitter)
  stream = StreamingNode(client, protocol=args.stream_protocol, port=args.stream_port, session=args.stream_session,user=args.stream_user, password=args.stream_password)
  saving = SaveNode(client, storage=args.save_storage, duration=args.save_duration, slice_time=args.save_slice_time)
  
  # Connect nodes
  camera.sink(model)  # connect camera to model
  camera.sink(stream)  # connect camera to stream
  camera.sink(saving)  # connect camera to saving
  
  # Set message handlers
  camera.onReceive = camera_message_handler
  model.onReceive = model_message_handler
  stream.onReceive = stream_message_handler
  saving.onReceive = save_message_handler
  
  # Start the client
  client.start(args.broker, args.port)
  
  print("Enter any key to exit...")
  
  try:
    input()
  finally:
    client.stop()


if __name__ == "__main__":
    main()
