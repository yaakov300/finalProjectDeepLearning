function start() {

    var result = undefined;
    $("#import").click(function() {
        if (!!result) {
            $.ajax({
                url: "/train/start/",
                type: "POST",
                contentType: 'application/json; charset=utf-8',
                data: JSON.stringify(result),

                success: function(result) {
                    $("#result").val(result);
                    $('.networks').css('display','none');
                    $('#networkDetails').css('display','block');
                }
            });
        }else{
            alert('import file first!!');
        }
    });

    $("#selectFiles").change(function(e) {
        onChange(e);
    });

    $('#start').on('click',function(){
        $("#result").val("started training wait for redirect to dashboard page its will take few second");
        setTimeout(function(){
            window.location = '/';
        },4000);

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
