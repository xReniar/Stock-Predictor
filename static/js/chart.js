let path = window.location.pathname;
let pathParts = path.split("/");
let stock = pathParts[pathParts.length - 1];

fetch(`/models`)
    .then(response => response.json())
    .then(models => {
        /*
        let results = [];
        models.forEach((model) => {
            fetch(`/prediction/${model}/${stock}`)
                .then(response => response.json())
                .then(data => {
                    var value = document.getElementById(`${model}-value`);
                    value.textContent = parseFloat(data).toFixed(4);
                    results.append(value);
                });
            fetch(`/mse/${model}/${stock}`)
                .then(response => response.json())
                .then(data => {
                    var mse = document.getElementById(`${model}-mse`);
                    mse.textContent = parseFloat(data).toFixed(4);
                });
        });
        */
        let predictionsPromises = models.map(async model => {
            const [prediction] = await Promise.all([
                fetch(`/prediction/${model}/${stock}`)
                    .then(response => response.json())
                    .then(data => {
                        var value = document.getElementById(`${model}-value`);
                        value.textContent = parseFloat(data).toFixed(4);
                        return {
                            "model": model,
                            "value": parseFloat(data).toFixed(4)
                        };
                    }),
                fetch(`/mse/${model}/${stock}`)
                    .then(response_1 => response_1.json())
                    .then(data_1 => {
                        var mse = document.getElementById(`${model}-mse`);
                        mse.textContent = parseFloat(data_1).toFixed(4);
                    })
            ]);
            return prediction;
        });

        return Promise.all(predictionsPromises);
    })
    .then((results) => { // approssima alcune volte
        fetch(`/chart_values/${stock}`)
            .then(response => response.json())
            .then(axis => {
                const stock_chart = document.getElementById("stock-chart");

                let colors = [
                    "rgba(255,0,0,1.0)",
                    "rgba(135,206,235,255)",
                    "rgba(0,255,0,1.0)",
                    "rgba(255,128,0,1.0)"
                ];

                let datasets = [
                    {
                        type: 'line',
                        label: stock,
                        data: axis.yValues,
                        backgroundColor: "rgba(0,0,0,1.0)",
                        borderColor: "rgba(0,0,0,1.0)",
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        type: 'line',
                        label: 'price',
                        data: new Array(axis.yValues.length).fill(axis.yValues.at(-1)),
                        backgroundColor: "rgba(127,127,127,1.0)",
                        borderColor: "rgba(127,127,127,1.0)",
                        pointRadius: 0
                    }
                ]
                for(let i = 0;i < results.length;i++){
                    datasets.push({
                        type: 'line',
                        label: results[i]["model"],
                        data: new Array(axis.yValues.length).fill(results[i]["value"]),
                        backgroundColor: colors[i],
                        borderColor: colors[i],
                        borderWidth: 1.3,
                        pointRadius: 0
                    })
                }

                const mixedChart = new Chart(stock_chart, {
                    data: {
                        datasets: datasets,
                        labels: axis.xValues
                    },
                });
            });
        })




/*
        new Chart(stock_chart, {
            type: "line",
            data: {
                labels: dataset["xValues"],
                datasets: [{
                    label: stock,
                    backgroundColor: "rgba(0,0,255,1.0)",
                    borderColor: "rgba(0,0,255,1.0)",
                    data: dataset["yValues"],
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                plugins: {
                    annotation: {
                        annotations: {
                            price: {
                                type: 'line',
                                yMin: dataset.yValues.at(-1),
                                yMax: dataset.yValues.at(-1),
                                borderColor: 'rgb(255, 99, 132)',
                                borderWidth: 2,
                            }
                        }
                    }
                }
            }
        });
*/