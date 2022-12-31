using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Tutorial
{
    class Program
    {
        private class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
        }

        private class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }

        public static void Main()
        {
            MLContext mlContext = new MLContext(1);

            // 1. Import or create training data
            HouseData[] houseData =
            {
                new HouseData() { Size = 1.1F, Price = 1.2F },
                new HouseData() { Size = 1.9F, Price = 2.3F },
                new HouseData() { Size = 2.8F, Price = 3.0F },
                new HouseData() { Size = 3.4F, Price = 3.7F }
            };

            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            // Define data preparation estimator
            IEstimator<ITransformer> dataPrepEstimator = mlContext.Transforms.Concatenate(
                "Features",
                new[] { "Size" }
            );

            // Create data preparation transformer
            ITransformer dataPrepTransformer = dataPrepEstimator.Fit(trainingData);

            // Pre-process data using data prep operations
            IDataView transformedData = dataPrepTransformer.Transform(trainingData);

            // 2. Specify model trainer
            // Define regression algorithm estimator
            OnlineGradientDescentTrainer? regressionEstimator =
                mlContext.Regression.Trainers.OnlineGradientDescent(
                    labelColumnName: "Price",
                    numberOfIterations: 100
                );
            // Create algorithm preparation transformer
            var regressionTransformer = regressionEstimator.Fit(transformedData);

            // 3. Create model
            var model = dataPrepTransformer.Append(regressionTransformer);

            // 4. Make a prediction
            var size = new HouseData() { Size = 2.5F };
            var price = mlContext.Model
                .CreatePredictionEngine<HouseData, Prediction>(model)
                .Predict(size);

            Console.WriteLine(
                $"//4 Predicted price for size: {size.Size * 1000} sq ft= {price.Price * 100:C}k"
            );

            // 5. Save model and and data transformer
            string regressionTransformerlPath = "./models/TutorialRegressionTransformer.zip";
            string dataTransformerPath = "./models/TutorialDataTransformer.zip";

            // Save Data Prep Transformer
            mlContext.Model.Save(dataPrepTransformer, trainingData.Schema, dataTransformerPath);
            // Save Trained Model
            mlContext.Model.Save(
                regressionTransformer,
                transformedData.Schema,
                regressionTransformerlPath
            );

            // 5.1 (OPTIONAL) Load model and data transformer
            // Define DataViewSchema of data prep pipeline and trained model
            DataViewSchema dataPrepPipelineSchema,
                modelSchema;

            // Load data preparation pipeline
            dataPrepTransformer = mlContext.Model.Load(
                dataTransformerPath,
                out dataPrepPipelineSchema
            );

            // Load trained model
            regressionTransformer =
                (RegressionPredictionTransformer<LinearRegressionModelParameters>)
                    mlContext.Model.Load(regressionTransformerlPath, out modelSchema);

            // 6. Get new data (repeat step 1)
            HouseData[] newDataSet =
            {
                new HouseData { Size = 4F, Price = 3F },
                new HouseData { Size = 2F, Price = 2.3F }
            };

            IDataView newTrainingData = mlContext.Data.LoadFromEnumerable(newDataSet);

            // Preprocess Data
            IDataView transformedNewData = dataPrepTransformer.Transform(newTrainingData);

            // 7. Extract trained model parameters
            // model.Last() is used here to extract weights and biases from model
            // as it is added to the end of the chain
            var originalModelParameters =
                ((ISingleFeaturePredictionTransformer<object>)model.Last()).Model
                as LinearRegressionModelParameters;

            var retrainedRegressionTransformer = regressionEstimator.Fit(
                transformedNewData,
                originalModelParameters
            );

            // 8. Recreate model
            model = dataPrepTransformer.Append(retrainedRegressionTransformer);

            // 9 Make a prediction on re-trained model
            var newPrice = mlContext.Model
                .CreatePredictionEngine<HouseData, Prediction>(model)
                .Predict(size);

            Console.WriteLine(
                $"//9 Predicted price for size: {size.Size * 1000} sq ft= {newPrice.Price * 100:C}k"
            );

            // The output is the following:
            //4 Predicted price for size: 2500 sq ft= £276.98k
            //9 Predicted price for size: 2500 sq ft= £228.00k
        }
    }
}
