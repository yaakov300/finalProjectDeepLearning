$(document).ready(function(){
    selected = $('#network_select').find(':selected').text();

    $("[name="+selected+"]").css('display','block');
    $('#network_select').on('change',function(){
        $('.layers').removeAttr('style');
        $("[name="+selected+"]").css('display','block');
    });

     for (i=1;i<=20;i++){
        $("#select-step").append($('<option></option>').val(i).html(i))
    }


    //ajax
    $('#uploadimage').on('submit',(function(e) {
        e.preventDefault();
        $('.loader_visual').show();
        var formData = new FormData($('#uploadimage').get(0));
        network_selected = $('#network_select').find(':selected').text();
        formData.append('network_name', network_selected);
        layer_selected = $('#layer_select').find(':selected').text();
        formData.append('layer_select', layer_selected);
        select_step = $('#select-step').find(':selected').text();
        formData.append('select-step', select_step);

        console.log(formData)
        $.ajax({
            type:'POST',
            url: '/visualition/test/',
            data:formData,
            cache:false,
            contentType: false,
            processData: false,
            success:function(data){
                console.log("success");
//                console.log(data);
                $("#image_place").empty();
                $("#image_place").append(data)
                $('.loader_visual').hide();
            },
            error: function(data){
                console.log("error");

                $('.loader_visual').hide()
            }
        });
//        $("#selectFiles").val("");

    }));

    $("#selectFiles").on("change", function() {
        console.log('change')
//        $("#imageUploadForm").submit();
        readURL(this);
    });

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#image').attr('src', e.target.result);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
//    $('#image_place').append("<img src=\"{% static 'vis/01804.jpg' %}\" style=\"width:100%; max-height:100%;\">")

});