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
});