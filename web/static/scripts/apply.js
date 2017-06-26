$(document).ready(function (e) {
    $('#uploadimage').on('submit',(function(e) {
        e.preventDefault();
        var formData = new FormData($('#uploadimage').get(0));
        network_selected = $('#network_select').find(':selected').text();
        formData.append('network_name', network_selected);

        console.log(formData)
        $.ajax({
            type:'POST',
            url: '/apply/test/',
            data:formData,
            cache:false,
            contentType: false,
            processData: false,
            success:function(data){
                console.log("success");
                console.log(data);
            },
            error: function(data){
                console.log("error");
                console.log(data);
            }
        });
//        $("#selectFiles").val("");

    }));

    $("#selectFiles").on("change", function() {
        console.log('change')
        $("#imageUploadForm").submit();
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

});