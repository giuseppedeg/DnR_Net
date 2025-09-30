import json
import base64
import os
from PIL import Image
from io import BytesIO

EXT = ".pap"

class FFHandler:
    def __init__(self):
        self.name = ""
        self.image_base64 = ""
        self.image_format = ""
        self.json_data = {}

    def save_formattedfile(self, image_path, json_path, output_path):
        """
        Save the file contatining the image and the json metadata
        """
        if not os.path.isdir(output_path):
            raise FileNotFoundError(f" {output_path} is not a folder.")

        with open(image_path, "rb") as img_file:
            self.image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        with open(json_path, "rb") as json_file:
            self.json_data =  base64.b64encode(json_file.read()).decode("utf-8")

        # I can load the metadata without b64 encoding. The file is a bit smaller
        # with open(json_path, "r") as json_file:
        #     self.json_data = json.load(json_file)
        
        self.name, self.image_format = os.path.splitext(os.path.basename(image_path))

        data = {"name": self.name,
                "img_format": self.image_format,
                "image": self.image_base64,
                "metadata": self.json_data,
                }

        with open(os.path.join(output_path, self.name+EXT), "w") as json_file:
            json.dump(data, json_file)


    def load_formattedfile(self, file_path):
        """
        Loads the file and return the image and the json metadata.

        Returns the PIL image and the metadata JSON
        """
        
        data = self.load_coded_formattedfile(file_path)

        image, json_data, name, image_format = self.decode_formattedfile(data)

        return image, json_data, name, image_format
    
    
    def load_coded_formattedfile(self, file_path):
        """
        Loads the file and return b64 coded data.

        Returns the PIL image and the metadata JSON
        """
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        return data
    

    def decode_formattedfile(self, data):
        """
        Decodes the data and return the image and the json metadata.

        Returns the PIL image and the metadata JSON
        """
        
        self.json_data = data["metadata"]
        self.image_base64 = data["image"]
        self.name = data["name"]
        self.image_format = data["img_format"]

        image_data = base64.b64decode(self.image_base64)
        image = Image.open(BytesIO(image_data))

        json_data = base64.b64decode(self.json_data).decode("utf-8")
        json_data = json.loads(json_data)

        return image, json_data, self.name, self.image_format

    
    def extract_info(self, file_path, out_dir):
        """
        Loads the file and extracts the image and the json metadata in the out_dir folder.
        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        image, json_data, _, _ = self.load_formattedfile(file_path)

        image.save(os.path.join(out_dir, self.name+self.image_format))

        with open(os.path.join(out_dir, self.name+".json"), "w") as json_file:
            json.dump(json_data, json_file, indent=4)


