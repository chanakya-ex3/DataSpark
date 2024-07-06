const express = require('express')
const app = express()
const axios = require('axios')


app.listen(3001, () => {
    console.log('Server is running on http://localhost:3000')
    })

app.get('/', (req, res) => {
  res.send('Hello World!')
}
)


app.get('/data', async (req, res) => {
    try {
      const response = await axios.get('http://127.0.0.1:8000/classificationInsights');
      res.json(response.data);
    } catch (error) {
      res.status(500).send('Error fetching data from FastAPI'+error);
    }
  });