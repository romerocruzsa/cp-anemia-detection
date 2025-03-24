from flask import Flask, request, jsonify
from HANDLER.images import DataHandler  # Correctly import the DataHandler class

app = Flask(__name__)

@app.route('/add_data', methods=['POST'])
def add_data():
    # Assuming you send the JSON file path via request or directly in the request body
    json_file_path = request.json.get('json_file_path')

    if json_file_path:
        data_handler = DataHandler(json_file_path)  # Create an instance of DataHandler
        result = data_handler.load_and_insert_data()
        return jsonify(result)
    else:
        return jsonify({"message": "Missing JSON file path!"})

@app.route('/get_data', methods=['GET'])
def get_data():
    data_handler = DataHandler('path_to_your_json_file.json')  # Example of creating an instance
    data_handler.fetch_data()
    return jsonify({"message": "Check the server logs for fetched data."})

if __name__ == "__main__":
    app.run(debug=True)
