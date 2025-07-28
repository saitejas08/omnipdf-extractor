#!/bin/bash

source .venv/bin/activate

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --full          Run complete pipeline (default)"
    echo "  --label-only    Only apply labeling to existing JSONs"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                    # Run full pipeline"
    echo "  ./run.sh --label-only       # Only label existing files"
}

# Parse command line arguments
MODE="full"
if [ "$1" = "--label-only" ]; then
    MODE="label-only"
elif [ "$1" = "--help" ]; then
    show_usage
    exit 0
elif [ "$1" != "" ] && [ "$1" != "--full" ]; then
    echo "❌ Unknown option: $1"
    show_usage
    exit 1
fi

echo "🚀 Starting PDF processing pipeline..."

if [ "$MODE" = "full" ]; then
    # Check if inputs directory exists for full pipeline
    if [ ! -d "inputs" ]; then
        echo "❌ 'inputs' directory not found. Please create it and add PDF files."
        exit 1
    fi

    # Check if there are PDF files
    pdf_count=$(find inputs -name "*.pdf" | wc -l)
    if [ $pdf_count -eq 0 ]; then
        echo "❌ No PDF files found in 'inputs' directory."
        exit 1
    fi

    echo "📄 Found $pdf_count PDF file(s) to process"
    echo "⚙️  Running complete pipeline (extraction + labeling)..."
    
    # Create outputs directory if it doesn't exist
    mkdir -p outputs
    
    # Run full pipeline
    python3 src/main.py
    
elif [ "$MODE" = "label-only" ]; then
    # Check if outputs directory exists
    if [ ! -d "outputs" ]; then
        echo "❌ 'outputs' directory not found. Nothing to label."
        exit 1
    fi

    # Check if there are JSON files
    json_count=$(find outputs -name "*.json" | wc -l)
    if [ $json_count -eq 0 ]; then
        echo "❌ No JSON files found in 'outputs' directory."
        exit 1
    fi

    echo "📄 Found $json_count JSON file(s) to label"
    echo "🏷️  Running labeling only..."
    
    # Run labeling only
    python3 src/main.py --label-only
fi

# Check if the pipeline completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Pipeline completed successfully!"
    echo "📁 Results saved in: outputs/"
    
    # Show summary of output files
    json_count=$(find outputs -name "*.json" | wc -l)
    echo "📊 Total JSON files: $json_count"
    
    # Show files with timestamps
    echo ""
    echo "📋 Generated files:"
    if [ $json_count -gt 0 ]; then
        ls -la outputs/*.json
    else
        echo "   No JSON files found"
    fi
    
else
    echo ""
    echo "❌ Pipeline failed with errors!"
    echo "💡 Check the logs above for details"
    exit 1
fi

echo ""
echo "🎉 All done! Check the 'outputs' directory for your processed files."
