from app import app ;

if __name__ == "__main__":
# Render sets the port dynamically, so bind to the PORT environment variable
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=False)
