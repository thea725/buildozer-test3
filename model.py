import numpy as np
from kivy.utils import platform

if platform == 'android':
    from jnius import autoclass

    File = autoclass('java.io.File')
    Interpreter = autoclass('org.tensorflow.lite.Interpreter')
    InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
    Tensor = autoclass('org.tensorflow.lite.Tensor')
    DataType = autoclass('org.tensorflow.lite.DataType')
    TensorBuffer = autoclass(
        'org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
    ByteBuffer = autoclass('java.nio.ByteBuffer')

    class TensorFlowModel():
        def load(self, model_filename, num_threads=None):
            model = File(model_filename)
            options = InterpreterOptions()
            if num_threads is not None:
                options.setNumThreads(num_threads)
            self.interpreter = Interpreter(model, options)
            self.allocate_tensors()

        def allocate_tensors(self):
            self.interpreter.allocateTensors()
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

        def get_input_shape(self):
            return self.input_shape

        def resize_input(self, shape):
            if self.input_shape != shape:
                self.interpreter.resizeInput(0, shape)
                self.allocate_tensors()

        def pred(self, x):
            # assumes one input and one output for now
            input = ByteBuffer.wrap(x.tobytes())
            output = TensorBuffer.createFixedSize(self.output_shape,
                                                  self.output_type)
            self.interpreter.run(input, output.getBuffer().rewind())
            return np.reshape(np.array(output.getFloatArray()),
                              self.output_shape)

elif platform == 'ios':
    from pyobjus import autoclass, objc_arr
    from ctypes import c_float, cast, POINTER

    NSString = autoclass('NSString')
    NSError = autoclass('NSError')
    Interpreter = autoclass('TFLInterpreter')
    InterpreterOptions = autoclass('TFLInterpreterOptions')
    NSData = autoclass('NSData')
    NSMutableArray = autoclass("NSMutableArray")

    class TensorFlowModel:
        def load(self, model_filename, num_threads=None):
            self.error = NSError.alloc()
            model = NSString.stringWithUTF8String_(model_filename)
            options = InterpreterOptions.alloc().init()
            if num_threads is not None:
                options.numberOfThreads = num_threads
            self.interpreter = Interpreter.alloc(
            ).initWithModelPath_options_error_(model, options, self.error)
            self.allocate_tensors()

        def allocate_tensors(self):
            self.interpreter.allocateTensorsWithError_(self.error)
            self.input_shape = self.interpreter.inputTensorAtIndex_error_(
                0, self.error).shapeWithError_(self.error)
            self.input_shape = [
                self.input_shape.objectAtIndex_(_).intValue()
                for _ in range(self.input_shape.count())
            ]
            self.output_shape = self.interpreter.outputTensorAtIndex_error_(
                0, self.error).shapeWithError_(self.error)
            self.output_shape = [
                self.output_shape.objectAtIndex_(_).intValue()
                for _ in range(self.output_shape.count())
            ]
            self.output_type = self.interpreter.outputTensorAtIndex_error_(
                0, self.error).dataType

        def get_input_shape(self):
            return self.input_shape

        def resize_input(self, shape):
            if self.input_shape != shape:
                # workaround as objc_arr doesn't work as expected on iPhone
                array = NSMutableArray.new()
                for x in shape:
                    array.addObject_(x)
                self.interpreter.resizeInputTensorAtIndex_toShape_error_(
                    0, array, self.error)
                self.allocate_tensors()

        def pred(self, x):
            # assumes one input and one output for now
            bytestr = x.tobytes()
            # must cast to ctype._SimpleCData so that pyobjus passes pointer
            floatbuf = cast(bytestr, POINTER(c_float)).contents
            data = NSData.dataWithBytes_length_(floatbuf, len(bytestr))
            print(dir(self.interpreter))
            self.interpreter.copyData_toInputTensor_error_(
                data, self.interpreter.inputTensorAtIndex_error_(
                    0, self.error), self.error)
            self.interpreter.invokeWithError_(self.error)
            output = self.interpreter.outputTensorAtIndex_error_(
                0, self.error).dataWithError_(self.error).bytes()
            # have to do this to avoid memory leaks...
            while data.retainCount() > 1:
                data.release()
            return np.reshape(
                np.frombuffer(
                    (c_float * np.prod(self.output_shape)).from_address(
                        output.arg_ref), c_float), self.output_shape)

else:
    import tensorflow as tf
    from six import BytesIO
    from PIL import Image
    import cv2

    class TensorFlowModel:
        def load(self, model_filename, num_threads=None):
            self.interpreter = tf.lite.Interpreter(model_filename,
                                                   num_threads=num_threads)
            self.interpreter.allocate_tensors()

        def detect_objects(self, image_path):
            # Load image and preprocess
            image = self.load_image_into_numpy_array(image_path)
            preprocessed_image = self.preprocess_image(image)

            # Set input tensor
            input_tensor_index = self.interpreter.get_input_details()[0]['index']
            self.interpreter.set_tensor(input_tensor_index, preprocessed_image)

            # Run inference
            self.interpreter.invoke()

            # Get output tensors
            output_details = self.interpreter.get_output_details()
            boxes = self.interpreter.get_tensor(output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(output_details[2]['index'])[0]

            # Filter out detections with low confidence
            threshold = 0.5
            selected_boxes = boxes[scores > threshold]
            selected_classes = classes[scores > threshold]

            return selected_boxes, selected_classes

        def load_image_into_numpy_array(self, path):
            # Load an image from file into a numpy array
            img_data = tf.io.gfile.GFile(path, 'rb').read()
            image = Image.open(BytesIO(img_data))
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

        def preprocess_image(self, image):
            # Preprocess the image before feeding it into the model
            image = cv2.resize(image, (320, 320))  # Adjust the size according to your model
            image = image / 255.0  # Normalize pixel values to the range [0, 1]
            image = np.float32(image)  # Convert to float32
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            return image

        def draw_boxes(self, image_path, boxes):
            # Draw bounding boxes on the image
            image = cv2.imread(image_path)
            for box in boxes:
                start_point = (int(box[1]), int(box[0]))
                end_point = (int(box[3]), int(box[2]))
                color = (0, 255, 0)  # Green color
                thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, thickness)
            return image