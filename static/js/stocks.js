// Funzione per creare e aggiungere un div per ogni valore di userData
function createStockDivs(stocks) {
    // Ottieni il body del documento
    var container = document.querySelector(".stocks-container");

    stocks.forEach((stock) => {
        // creating stock card header
        var card_header = document.createElement("div");
        card_header.className = "card-header py-3";
        var stock_name = document.createElement("h4");
        stock_name.className = "card-title pricing-card-title";
        stock_name.textContent = stock;
        card_header.appendChild(stock_name);

        // creating stock card body
        var card_body = document.createElement("div");
        card_body.className = "card-body";
        var stock_price = document.createElement("h1");
        stock_price.className = "my-0 fw-normal";
        stock_price.textContent = "Loading";
        setInterval(function() {
            fetch('/value/' + stock)
                .then(response => response.json())
                .then(data => {
                    fetch(`/price_state/${stock}`)
                        .then(response => response.json())
                        .then(data => {
                            if(data > 0){
                                stock_price.classList.add('text-success');
                                stock_price.classList.remove('text-danger');
                            } else {
                                stock_price.classList.add('text-danger');
                                stock_price.classList.remove('text-success');
                            }
                        })
                        .catch(error => {
                            consol.error("Error fetching data:", error);
                        });
                    stock_price.textContent = parseFloat(data).toFixed(4);
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                });
        }, 5000);
        card_body.appendChild(stock_price);

        // creating card div
        var card_div = document.createElement("div");
        card_div.className = "card mb-4 rounded-3 shadow-sm";
        card_div.appendChild(card_header);
        card_div.append(card_body);
        card_div.addEventListener("click",() => {
            window.location.href = `/chart/${stock}`;
        });

        // creating col div
        var col_div = document.createElement("div");
        col_div.className = "col";
        col_div.appendChild(card_div);

        container.appendChild(col_div);
    });
}

// Dati degli stocks passati dal server
fetch("/get_stocks")
    .then(response => response.json())
    .then(data => {
        /* creating the div for every stock */
        createStockDivs(data);
    })