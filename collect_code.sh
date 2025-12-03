#!/bin/bash

OUTPUT="project_code_dump.txt"

echo "⚙️ Generating project code dump from packages/, services/, and training/..."
echo "" > $OUTPUT

# Directories to include
INCLUDE_DIRS=("packages" "services" "training")

# Excluded patterns
EXCLUDES=(
    "__pycache__"
    "*.pyc"
    "*.gz"
    "*.ubyte"
    "*.pkl"
    "*.pt"
    "services/proto/*_pb2.py"
    "services/proto/*_pb2_grpc.py"
)

# Build base find command for included dirs
for DIR in "${INCLUDE_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        FIND_CMD="find $DIR -type f"

        # Apply excludes
        for EX in "${EXCLUDES[@]}"; do
            FIND_CMD+=" ! -path \"*/$EX\""
        done

        # Execute and dump contents
        eval $FIND_CMD | while read FILE; do
            # Only dump textual files
            if file "$FILE" | grep -q "text"; then
                echo "============================" >> $OUTPUT
                echo "File: $FILE" >> $OUTPUT   
                echo "============================" >> $OUTPUT
                cat "$FILE" >> $OUTPUT
                echo -e "\n\n" >> $OUTPUT
            fi
        done
    fi
done

echo "✅ Code exported to $OUTPUT"