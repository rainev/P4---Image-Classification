call .venv\Scripts\activate

set PROTO_DIR=.
set TRAINING_OUT=..\training
set DASHBOARD_OUT=..\dashboard
set PROTO_FILE=dashboard.proto

python -m grpc_tools.protoc -I=%PROTO_DIR% --python_out=%TRAINING_OUT% --grpc_python_out=%TRAINING_OUT% %PROTO_DIR%\%PROTO_FILE%
python -m grpc_tools.protoc -I=%PROTO_DIR% --python_out=%DASHBOARD_OUT% --grpc_python_out=%DASHBOARD_OUT% %PROTO_DIR%\%PROTO_FILE%

pause
