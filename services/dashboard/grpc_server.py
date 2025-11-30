# services/dashboard_app/grpc_server.py
from concurrent import futures
import time
import grpc

from services.proto import dashboard_pb2, dashboard_pb2_grpc


class DashboardService(dashboard_pb2_grpc.DashboardServiceServicer):
    def __init__(self, state_manager):
        self.state = state_manager

    def StreamTrainingData(self, request_iterator, context):
        """
        Receives a stream of TrainingBatch messages from the training app.
        For each batch, we update the shared state for the UI to consume.
        """
        try:
            for batch in request_iterator:
                self.state.update_from_batch(batch)
            return dashboard_pb2.Ack(ok=True, message="Training stream ended.")
        except grpc.RpcError as e:
            # You can log this and mention it in fault-tolerance discussion
            return dashboard_pb2.Ack(ok=False, message=f"RPC error: {e}")

    def Ping(self, request, context):
        """
        Health check RPC. Returns current FPS and queue size, plus server time.
        """
        fps, queue_size = self.state.get_status()
        now_ms = int(time.time() * 1000)
        return dashboard_pb2.PingResponse(
            alive=True,
            status="OK",
            server_timestamp_ms=now_ms,
            frames_per_second=int(fps),
            queue_size=queue_size,
        )


def serve_in_foreground(state_manager, port: int = 50051):
    """
    Starts the gRPC server and blocks. Intended to be called inside a background
    thread from main.py so the UI thread remains free.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(
        DashboardService(state_manager),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[Dashboard] gRPC server started on port {port}")
    server.wait_for_termination()