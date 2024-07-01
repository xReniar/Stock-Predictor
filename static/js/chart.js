let path = window.location.pathname;
let pathParts = path.split("/");
let stock = pathParts[pathParts.length - 1];

fetch(`/chart_values/${stock}`)
    .then(response => response.json())
    .then(dataset => {
        const stock_chart = document.getElementById("stock-chart");

        new Chart(stock_chart, {
            type: "line",
            data: {
                labels: dataset["xValues"],
                datasets: [{
                    label: stock,
                    backgroundColor:"rgba(0,0,255,1.0)",
                    borderColor: "rgba(0,0,255,0.1)",
                    data: dataset["yValues"],
                    tension: 0.4
                }]
            },
        });
        
    });