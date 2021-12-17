import express from 'express';
import fileUpload from 'express-fileupload';
import controller from './image-classification/controller';
import path from 'path';

const app = express();
app.use(fileUpload());
app.use(express.static(path.join(__dirname, 'public')))

app.post('/classify', controller.classifyImage);

export default app;
