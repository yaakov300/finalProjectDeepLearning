function start() {

    var result = undefined;
    $("#import").click(function() {
        if (!!result) {
            $.ajax({
                url: "/home/",
                type: "POST",
                data: result,
                success: function(result) {
                    $("#result").val(result);
                }
            });
        }
    });

    $("#selectFiles").change(function(e) {
        onChange(e);
    });

    function onChange(event) {
        var reader = new FileReader();
        reader.onload = onReaderLoad;
        reader.readAsText(event.target.files[0]);
        console.log(event.target.files[0]);
    }

    function onReaderLoad(event){
        //alert(event.target.result);
        $('#json_content').val(event.target.result);
        result = JSON.parse(event.target.result);
        console.log('json load');
    }
}

$(document).ready(function(){
    start();
});
