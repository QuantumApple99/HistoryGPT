function process() {
    var prompt = document.getElementById("message-input").value;
    new_prompt = "User: " + prompt + "\n";
    document.getElementById("messages").innerHTML = new_prompt;
    document.getElementById("message-input").value = "";
    console.log("Successful");

    $.ajax({
        type: "POST",
        url: "/process_data",
        data: JSON.stringify({ prompt: prompt }),
        contentType: "application/json; charset=utf-8",
        success: function (response) {
            text = new_prompt + "\n\n" + "HistoryGPT: " + response.message;
            document.getElementById("messages").innerHTML = text;
        }
    });
}